# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from collections.abc import Generator
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.lazy import torch
from anemoi.inference.metadata import Metadata
from anemoi.inference.profiler import ProfilingLabel
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from . import runner_registry

LOG = logging.getLogger(__name__)


class TemporalDownscalerMetadata(Metadata):
    @cached_property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields.
        A temporal downscaler looks ahead instead of backwards.
        """
        return sorted([s * to_timedelta(self.data_frequency) for s in self.input_explicit_times])

    @cached_property
    def output_offsets(self) -> list[datetime.timedelta]:
        """Return the target times relative to the start of the temporal downscaling window."""
        start = self.input_explicit_times[0]
        return sorted([(t - start) * to_timedelta(self.data_frequency) for t in self.target_explicit_times])

    @cached_property
    def rollout_shift(self) -> datetime.timedelta:
        """Return the length of the temporal downscaling window."""
        return (self.input_explicit_times[-1] - self.input_explicit_times[0]) * to_timedelta(self.data_frequency)


@runner_registry.register("temporal_downscaler")
class TemporalDownscalerMultiOutRunner(Runner):
    """A runner to be used for inference of a trained temporal downscaler with multiple output steps.
    Unlike the single output, the temporal downscaling is all done as one step.
    Can be applied directly on analysis/forecast data without being coupled to a forecasting model.

    This runner makes the following assumptions:
        - The model was trained with two input states: (t and t+temporal_downscaling_window)
        - The output states are set by the checkpoint's target explicit times, and may include either boundary
    """

    def __init__(self, config: RunConfiguration, fill_timeline_with_input: bool = False):
        """Initialise the temporal downscaler runner.

        Parameters
        ----------
        config : RunConfiguration
            The run configuration.
        fill_timeline_with_input : bool, optional
            Write input states at window boundaries where the model does not output predictions, by default False.
        """
        super().__init__(
            config,
            classes=RunnerClasses(
                metadata=TemporalDownscalerMetadata,
            ),
        )

        self.fill_timeline_with_input = fill_timeline_with_input

        assert len(self.checkpoint.input_explicit_times) == 2, (
            "Temporal downscaler runner requires exactly two input explicit times (t and t+temporal_downscaling_window), "
            f"but got {self.checkpoint.input_explicit_times}"
        )

        self.temporal_downscaling_window = to_timedelta(self.checkpoint.data_frequency) * (
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0]
        )

        assert len(self.checkpoint.target_explicit_times) in (
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0] - 1,
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0],
        ), (
            "Temporal downscaler runner requires the number of target explicit times to be "
            "temporal_downscaling_window / frequency - 1 (no boundary) or temporal_downscaling_window / frequency "
            "(one boundary), but got "
            f"{len(self.checkpoint.target_explicit_times)} for temporal_downscaling_window {self.temporal_downscaling_window} and "
            f"input explicit times {self.checkpoint.input_explicit_times}"
        )

        self.lead_time = to_timedelta(self.config.lead_time)
        self.time_step = self.temporal_downscaling_window

        boundary_idx = self.checkpoint.input_explicit_times
        target_times = self.checkpoint.target_explicit_times
        self.left_boundary_predicted = boundary_idx[0] in target_times
        # A window's start is covered by the previous window's prediction (right boundary) or its own (left boundary)
        self.write_window_start = self.fill_timeline_with_input and not (
            self.left_boundary_predicted or boundary_idx[-1] in target_times
        )

        if self.write_window_start:
            for dataset, output in self.outputs.items():
                if not output.write_step_zero:
                    raise ValueError(
                        f"`fill_timeline_with_input` requires the initial state to be written, "
                        f"but `write_initial_state` is false for the output of dataset {dataset!r}."
                    )

    def patch_data_request(self, request: dict, dataset_name: str) -> dict:
        """Set sensible defaults when this runner is used with the `retrieve` command."""
        req = request.copy()

        req = super().patch_data_request(req, dataset_name)

        # by default the `time` will be two initialisation times, e.g. 0000 and 0600
        # instead, we want one initialisation time and use `step` to get the input forecast based on the lead time.
        if self.reference_date is not None:
            req["date"] = f"{self.reference_date.strftime('%Y-%m-%d')}"
            req["time"] = f"{self.reference_date.hour * 100:04d}"
        step_hours = int(self.temporal_downscaling_window.total_seconds() // 3600)
        lead_time_hours = int(self.lead_time.total_seconds() // 3600)
        req["step"] = list(range(0, lead_time_hours + step_hours, step_hours))
        return req

    def execute(self) -> None:
        """Execute the temporal downscaler runner with support for multiple temporal downscaling periods."""
        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        if self.reference_date is None:
            raise ValueError("Temporal downscaler runner requires a date set in the config or CLI.")

        # Calculate how many temporal downscaling windows we need for the lead_time
        num_windows = int(self.lead_time / self.temporal_downscaling_window)

        if self.lead_time % self.temporal_downscaling_window != to_timedelta(0):
            LOG.warning(
                "Lead time %s is not a multiple of temporal downscaling window %s. Will perform temporal downscaling for %s",
                self.lead_time,
                self.temporal_downscaling_window,
                num_windows * self.temporal_downscaling_window,
            )

        # create constant forcings states only once, they will be reused in each window
        self.constants_states: dict[str, State] = {}
        for dataset in self.tensor_handlers:
            self.constants_states[dataset] = self.constant_forcings_inputs[dataset].create_input_state(
                dates=[self.reference_date], constant=True, ref_date_index=0
            )
            for key in self.constants_states[dataset]["fields"].keys():
                self.constants_states[dataset]["fields"][key] = np.concatenate(
                    [
                        self.constants_states[dataset]["fields"][key],
                        self.constants_states[dataset]["fields"][key],
                    ],
                    axis=0,
                )
            self._check_state(dataset, self.constants_states[dataset], "constant_forcings")

        self.input_states_hook(self.constants_states)

        # Process each temporal downscaling window
        for window_idx in range(num_windows):
            window_start_date = self.reference_date + window_idx * self.temporal_downscaling_window
            window_end_date = window_start_date + self.temporal_downscaling_window

            LOG.info(
                f"Processing temporal downscaling window {window_idx + 1}/{num_windows} starting at {window_start_date}"
            )

            input_states: dict[str, State] = {}
            for dataset in self.tensor_handlers:
                prognostic_state = self.prognostics_inputs[dataset].create_input_state(
                    dates=[window_start_date, window_end_date], select_reference_date=True, ref_date_index=0
                )
                self._check_state(dataset, prognostic_state, "prognostics")

                forcings_state = self.dynamic_forcings_inputs[dataset].create_input_state(
                    dates=[window_start_date, window_end_date], select_reference_date=True, ref_date_index=0
                )
                self._check_state(dataset, forcings_state, "dynamic_forcings")

                self.constants_states[dataset]["date"] = window_start_date

                input_states[dataset] = self._combine_states(
                    prognostic_state,
                    self.constants_states[dataset],
                    forcings_state,
                )

                if window_idx == 0:
                    initial_state = input_states[dataset].copy()
                    for processor in self.post_processors[dataset]:
                        initial_state = processor.process(initial_state)

                    self.outputs[dataset].open(initial_state)

            # Run temporal downscaler for this window
            for states in self.run(input_states=input_states, lead_time=self.temporal_downscaling_window):
                for dataset, state in states.items():
                    if not state["temporally_downscaled"]:
                        if window_idx == 0 and self.left_boundary_predicted:
                            continue
                        if window_idx > 0 and not self.write_window_start:
                            continue

                    is_input_step_zero = window_idx == 0 and not state["temporally_downscaled"]

                    # Updating state step to be a global step not relative to window
                    state["step"] = state["step"] + window_idx * self.temporal_downscaling_window

                    # Apply post-processing
                    for processor in self.post_processors[dataset]:
                        state = processor.process(state)

                    if is_input_step_zero:
                        self.outputs[dataset].write_initial_state(state)
                    else:
                        self.outputs[dataset].write_state(state)

        for output in self.outputs.values():
            output.close()

    def temporal_downscaler_stepper(
        self, start_date: datetime.datetime
    ) -> Generator[tuple[datetime.timedelta, datetime.datetime], None, None]:
        """Generate step and date variables for the temporal downscaling loop.

        Parameters
        ----------
        start_date : datetime.datetime
            Input start date.

        Returns
        -------
        step : datetime.timedelta
            Time delta between the target index date and the start date.
        date : datetime.datetime
            Valid date of the target index.
        """
        target_steps = self.checkpoint.target_explicit_times
        boundary_idx = self.checkpoint.input_explicit_times
        steps = len(target_steps)

        LOG.info(
            f"Time stepping: [{start_date}, {start_date + self.temporal_downscaling_window}], Temporal downscaling {steps} steps"
        )

        for target in target_steps:
            step = self.temporal_downscaling_window * (target - boundary_idx[0]) / (boundary_idx[-1] - boundary_idx[0])
            date = start_date + step
            yield step, date

    def forecast(
        self,
        lead_time: datetime.timedelta,
        input_tensors_numpy: dict[str, FloatArray],
        input_states: dict[str, State],
    ) -> Generator[dict[str, State], None, None]:
        """Temporally downscale between the current and future state in the input tensor.

        Parameters
        ----------
        lead_time : datetime.timedelta
            Unused. This method processes one temporal downscaling window at a time.
        input_tensors_numpy : dict[str, FloatArray]
            The input tensors for each dataset, as numpy arrays with shape (multi_step_input, variables, values).
        input_states : dict[str, State]
            The input states for each dataset. It contains both input dates defined by the config explicit_times.input.

        Returns
        -------
        dict[str, State]
            The temporally downscaled states for each dataset.
        """
        # This does temporal downscaling but called forecast so we can reuse run()
        with torch.inference_mode():
            self.model.eval()

            # Create pytorch input tensor
            input_tensors_torch = {
                dataset: torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)
                for dataset, input_tensor_numpy in input_tensors_numpy.items()
            }

            LOG.info("Using autocast %s", self.autocast)

            new_states = input_states.copy()  # We should not modify the input state
            for dataset, handler in self.tensor_handlers.items():
                new_states[dataset]["fields"] = dict()
                new_states[dataset]["step"] = to_timedelta(0)
                start = input_states[dataset]["date"]

                if self.verbosity > 0:
                    handler._print_input_tensor("First input tensor", input_tensors_torch[dataset])

            # First yield the boundary states (t and t+window_size)
            boundary_times = self.checkpoint.input_explicit_times

            initial_states: dict[str, State] = {}
            for dataset, handler in self.tensor_handlers.items():
                # Yield initial boundary state (t)
                initial_states[dataset] = new_states[dataset].copy()
                initial_states[dataset]["date"] = start
                initial_states[dataset]["step"] = to_timedelta(0)
                initial_states[dataset]["temporally_downscaled"] = False
                # Extract fields from the first time step of input tensor
                input_numpy = (
                    input_tensors_torch[dataset][0, boundary_times[0]].cpu().numpy()
                )  # # Select the initial boundary state (t) - First time step
                for i in range(input_numpy.shape[-1]):
                    var_name = None
                    for (
                        var,
                        idx,
                    ) in handler.metadata.variable_to_input_tensor_index.items():
                        if idx == i:
                            var_name = var
                            break
                    if var_name and var_name in handler.metadata.output_tensor_index_to_variable.values():
                        initial_states[dataset]["fields"][var_name] = input_numpy[:, i]

            yield initial_states

            # Predict next state of atmosphere
            with (
                torch.autocast(device_type=self.device.type, dtype=self.autocast),
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(f"Temporal downscaling step ({start})"),
            ):
                y_pred = self.predict_step(self.model, input_tensors_torch, fcstep=0)

            # Now perform temporal downscaling between the boundaries
            for s, (step, date) in enumerate(self.temporal_downscaler_stepper(start)):
                for dataset, handler in self.tensor_handlers.items():
                    new_states[dataset]["date"] = date
                    new_states[dataset]["previous_step"] = new_states[dataset].get("step")
                    new_states[dataset]["step"] = step
                    new_states[dataset]["temporally_downscaled"] = True

                    if handler.trace:
                        handler.trace.write_input_tensor(
                            date,
                            s,
                            input_tensors_torch[dataset].cpu().numpy(),
                            handler.metadata.variable_to_input_tensor_index,
                            self.checkpoint.timestep,
                        )

                    # Detach tensor and squeeze (should we detach here?)
                    with ProfilingLabel("Sending output to cpu", self.use_profiler):
                        # squeeze batch and ensemble dims of tensor with shape: (batch, steps, ensemble, values, variables)
                        output = np.squeeze(
                            y_pred[dataset].cpu().numpy(), axis=(0, 2)
                        )  # shape: (steps, values, variables)

                    if handler.trace:
                        handler.trace.write_output_tensor(
                            date,
                            s,
                            output[s],
                            handler.metadata.output_tensor_index_to_variable,
                            self.checkpoint.timestep,
                        )

                    # Update state
                    with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                        for i in range(output.shape[2]):
                            new_states[dataset]["fields"][handler.metadata.output_tensor_index_to_variable[i]] = output[
                                s, :, i
                            ]

                    if self.verbosity > 0:
                        handler._print_output_tensor("Output tensor", output)

                yield new_states
