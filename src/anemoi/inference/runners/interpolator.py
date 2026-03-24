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
from anemoi.inference.forcings import ConstantDateForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.lazy import torch
from anemoi.inference.metadata import Metadata
from anemoi.inference.profiler import ProfilingLabel
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.tensors import Kind
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from . import runner_registry

LOG = logging.getLogger(__name__)


class TimeInterpolatorTensorHandler(TensorHandler):
    def add_initial_forcings_to_input_state(self, input_state: State) -> None:
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.metadata.lagged]
        reference_date = self.context.reference_date or date
        reference_dates = [reference_date + h for h in self.metadata.lagged]

        # TODO: Check for user provided forcings

        # We may need different forcings initial conditions
        initial_constant_forcings_inputs = self.context.initial_constant_forcings_inputs(self.constant_forcings_inputs)
        initial_dynamic_forcings_inputs = self.context.initial_dynamic_forcings_inputs(self.dynamic_forcings_inputs)

        LOG.info("-" * 80)
        LOG.info("Initial forcings inputs:")
        LOG.info("  Constant forcings:")
        for f in initial_constant_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("  Dynamic forcings:")
        for f in initial_dynamic_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("Initial forcings dates:")
        LOG.info(f"  {', '.join([date.isoformat() for date in dates])}")
        LOG.info("-" * 80)

        for source in initial_constant_forcings_inputs:
            arrays = source.load_forcings_array(reference_dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial constant forcings")

        for source in initial_dynamic_forcings_inputs:
            arrays = source.load_forcings_array(dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=False, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial dynamic forcings")

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = ConstantDateForcings(self, self.constant_forcings_input, variables, mask)
        return [result]


class TimeInterpolatorMetadata(Metadata):
    @cached_property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields.
        An interpolator looks ahead instead of backwards.
        """
        return sorted([s * to_timedelta(self.data_frequency) for s in self.input_explicit_times])


@runner_registry.register("time_multi_interpolator")
class TimeInterpolatorMultiOutRunner(Runner):
    """A runner to be used for inference of a trained interpolator with multiple output steps.
    Unlike the single output, the interpolation is all done as one step.
    Can be applied directly on analysis/forecast data without being coupled to a forecasting model.

    This runner makes the following assumptions:
        - The model was trained with two input states: (t and t+interpolation_window)
        - The output states are between these two states and are set by "frequency" in the config
    """

    def __init__(self, config: RunConfiguration):
        super().__init__(
            config,
            classes=RunnerClasses(tensor_handler=TimeInterpolatorTensorHandler, metadata=TimeInterpolatorMetadata),
        )

        self.multi_step_input = 2

        assert len(self.checkpoint.input_explicit_times) == 2, (
            "Interpolator runner requires exactly two input explicit times (t and t+interpolation_window), "
            f"but got {self.checkpoint.input_explicit_times}"
        )
        assert len(self.checkpoint.target_explicit_times) in (
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0] - 1,
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0],
        ), (
            "Interpolator runner requires the number of target explicit times to be equal to "
            "interpolation_window / frequency - 1, but got "
            f"{len(self.checkpoint.target_explicit_times)} for interpolation_window {self.interpolation_window} and "
            f"input explicit times {self.checkpoint.input_explicit_times}"
        )

        self.interpolation_window = to_timedelta(self.checkpoint.data_frequency) * (
            self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0]
        )
        self.lead_time = to_timedelta(self.config.lead_time)
        self.time_step = self.interpolation_window

    def patch_data_request(self, request: dict, dataset_name: str) -> dict:
        """Set sensible defaults when this runner is used with the `retrieve` command."""
        req = request.copy()

        req = super().patch_data_request(req, dataset_name)

        # by default the `time` will be two initialisation times, e.g. 0000 and 0600
        # instead, we want one initialisation time and use `step` to get the input forecast based on the lead time.
        if self.reference_date is not None:
            req["time"] = f"{self.reference_date.hour*100:04d}"
        req["step"] = (
            f"0/to/{int(self.lead_time.total_seconds()//3600)}/by/{int(self.interpolation_window.total_seconds()//3600)}"
        )
        return req

    def execute(self) -> None:
        """Execute the interpolator runner with support for multiple interpolation periods."""
        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        # Get the interpolation window size from training config
        boundary_idx = self.checkpoint.input_explicit_times
        # Calculate how many interpolation windows we need for the lead_time
        num_windows = int(self.lead_time / self.interpolation_window)

        if self.lead_time % self.interpolation_window != to_timedelta(0):
            LOG.warning(
                "Lead time %s is not a multiple of interpolation window %s. Will interpolate for %s",
                self.lead_time,
                self.interpolation_window,
                num_windows * self.interpolation_window,
            )

        # create constant forcings states only once, they will be reused in each window
        self.constants_states: dict[str, State] = {}
        for dataset in self.tensor_handlers:
            self.constants_states[dataset] = self.constant_forcings_inputs[dataset].create_input_state(
                date=self.config.date, constant=True, ref_date_index=0
            )
            for key in self.constants_states[dataset]["fields"].keys():
                self.constants_states[dataset]["fields"][key] = np.concatenate(
                    [self.constants_states[dataset]["fields"][key], self.constants_states[dataset]["fields"][key]],
                    axis=0,
                )
            self._check_state(self.constants_states[dataset], "constant_forcings")

        # Process each interpolation window
        for window_idx in range(num_windows):
            window_start_date = self.config.date + window_idx * self.interpolation_window

            LOG.info(f"Processing interpolation window {window_idx + 1}/{num_windows} starting at {window_start_date}")

            input_states: dict[str, State] = {}
            for dataset in self.tensor_handlers:
                prognostic_state = self.prognostics_inputs[dataset].create_input_state(
                    date=window_start_date, select_reference_date=True, ref_date_index=0
                )
                self._check_state(prognostic_state, "prognostics")

                forcings_state = self.dynamic_forcings_inputs[dataset].create_input_state(
                    date=window_start_date, select_reference_date=True, ref_date_index=0
                )
                self._check_state(forcings_state, "dynamic_forcings")

                self.constants_states[dataset]["date"] = window_start_date

                input_states[dataset] = self._combine_states(
                    prognostic_state,
                    self.constants_states[dataset],
                    forcings_state,
                )

                self.input_state_hook(self.constants_states[dataset])

                initial_state = input_states[dataset].copy()
                for processor in self.post_processors[dataset]:
                    initial_state = processor.process(initial_state)

                self.outputs[dataset].open(initial_state)

            # Run interpolation for this window
            for state_idx, states in enumerate(
                self.run(input_states=input_states, lead_time=self.interpolation_window)
            ):
                for dataset, state in states.items():
                    # In the first window, we want to write the initial state (t=0)
                    # In other windows, we want to skip the initial state (t=0)
                    # because it is written as the last state of the previous window
                    if window_idx != 0 and state_idx == boundary_idx[0]:
                        continue

                    # Updating state step to be a global step not relative to window
                    state["step"] = state["step"] + window_idx * self.interpolation_window

                    # Apply post-processing
                    for processor in self.post_processors[dataset]:
                        state = processor.process(state)

                    self.outputs[dataset].write_state(state)

        for output in self.outputs.values():
            output.close()

    def interpolator_stepper(
        self, start_date: datetime.datetime
    ) -> Generator[tuple[datetime.timedelta, datetime.datetime], None, None]:
        """Generate step and date variables for the interpolation loop.

        Parameters
        ----------
        start_date : datetime.datetime
            Input start date

        Returns
        -------
        step : datetime.timedelta
            Time delta between the target index date and the start date
        date : datetime.datetime
            Date of the zeroth index of the input tensor
        """
        target_steps = self.checkpoint.target_explicit_times
        boundary_idx = self.checkpoint.input_explicit_times
        steps = len(target_steps)

        LOG.info(
            f"Time stepping: [{start_date}, {start_date + self.interpolation_window}], Interpolating {steps} steps"
        )

        for s in range(steps):
            step = self.interpolation_window * (s + 1) / (boundary_idx[-1] - boundary_idx[0])
            date = start_date + step
            yield step, date

    def forecast(
        self, lead_time: datetime.timedelta, input_tensors_numpy: dict[str, FloatArray], input_states: dict[str, State]
    ) -> Generator[dict[str, State], None, None]:
        """Interpolate between the current and future state in the input tensor.

        Parameters
        ----------
        lead_time : datetime.timedelta
            Unused. This method processes one interpolation window at a time.
        input_tensors_numpy : dict[str, FloatArray]
            The input tensors for each dataset, as numpy arrays with shape (multi_step_input, variables, values).
        input_states : dict[str, State]
            The input states for each dataset. It contains both input dates defined by the config explicit_times.input

        Returns
        -------
        dict[str, State]
            The interpolated states for each dataset.
        """
        # This does interpolation but called forecast so we can reuse run()
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

            if self.write_initial_state:
                initial_states: dict[str, State] = {}
                for dataset, handler in self.tensor_handlers.items():
                    # Yield initial boundary state (t)
                    initial_states[dataset] = new_states[dataset].copy()
                    initial_states[dataset]["date"] = start
                    initial_states[dataset]["step"] = to_timedelta(0)
                    initial_states[dataset]["interpolated"] = False
                    # Extract fields from the first time step of input tensor
                    input_numpy = (
                        input_tensors_torch[dataset][0, boundary_times[0]].cpu().numpy()
                    )  # # Select the initial boundary state (t) - First time step
                    for i in range(input_numpy.shape[-1]):
                        var_name = None
                        for var, idx in handler.metadata.variable_to_input_tensor_index.items():
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
                Timer(f"Interpolating step ({start})"),
            ):
                y_pred = self.predict_step(self.model, input_tensors_torch)

            # Now interpolate between the boundaries
            for s, (step, date) in enumerate(self.interpolator_stepper(start)):
                for dataset, handler in self.tensor_handlers.items():
                    new_states[dataset]["date"] = date
                    new_states[dataset]["previous_step"] = new_states[dataset].get("step")
                    new_states[dataset]["step"] = step
                    new_states[dataset]["interpolated"] = True

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
                        output = np.squeeze(y_pred[dataset].cpu().numpy())  # shape: (values, variables)

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
