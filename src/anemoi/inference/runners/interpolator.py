# (C) Copyright 2024 Anemoi contributors.
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

import numpy as np
import torch
import torch.nn.functional
from anemoi.models.models import AnemoiModelEncProcDecInterpolator
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from numpy.typing import NDArray

from anemoi.inference.config import Configuration
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runner import Kind
from anemoi.inference.types import State

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..profiler import ProfilingLabel
from ..runners.default import DefaultRunner
from . import runner_registry

LOG = logging.getLogger(__name__)


def get_interpolation_window(data_frequency, input_explicit_times) -> datetime.timedelta:
    """Get the interpolation window."""
    return to_timedelta(data_frequency) * (input_explicit_times[1] - input_explicit_times[0])


def checkpoint_lagged_interpolator_patch(self) -> list[datetime.timedelta]:
    # For interpolator, we always want positive timedeltas
    result = [s * to_timedelta(self.data_frequency) for s in self.input_explicit_times]
    return sorted(result)


@runner_registry.register("time_interpolator")
class TimeInterpolatorRunner(DefaultRunner):
    """A runner to be used for inference of a trained interpolator directly on analysis data
    without being coupled to a forecasting model.
    """

    def __init__(self, config: Configuration) -> None:
        """Initialize the TimeInterpolatorRunner
        The runner makes the following assumptions:
            - The model was trained with two input states: (t and t+interpolation_window)
            - The output states are between these two states and are set by "frequency" in the config
            - interpolation_window / frequency - 1 is equal to the number of output states

        Parameters
        ----------
        config : Configuration | dict | str | BaseModel | None
            The configuration for the runner.
        **kwargs : dict
            Keyword arguments to initialize a config for the runner.
        """
        # assert config is not None or kwargs is not None, "Either config or kwargs must be provided"
        # config = config or kwargs

        # # Remove that when the Pydantic model is ready
        # if not isinstance(config, BaseModel):
        #     config = RunConfiguration.load(config)

        super().__init__(config)

        self.patch_checkpoint_lagged_property()
        assert (
            self.config.write_initial_state
        ), "Interpolator output should include temporal start state, end state and boundary conditions"
        assert isinstance(
            self.model.model, AnemoiModelEncProcDecInterpolator
        ), "Model must be an interpolator model for this runner"

        self.target_forcings = self.target_computed_forcings(
            self.checkpoint._metadata._config_training.target_forcing.data
        )

        assert len(self.checkpoint.input_explicit_times) == 2, (
            "Interpolator runner requires exactly two input explicit times (t and t+interpolation_window), "
            f"but got {self.checkpoint.input_explicit_times}"
        )
        assert (
            len(self.checkpoint.target_explicit_times)
            == self.checkpoint.input_explicit_times[1] - self.checkpoint.input_explicit_times[0] - 1
        ), (
            "Interpolator runner requires the number of target explicit times to be equal to "
            "interpolation_window / frequency - 1, but got "
            f"{len(self.checkpoint.target_explicit_times)} for interpolation_window {self.interpolation_window} and "
            f"input explicit times {self.checkpoint.input_explicit_times}"
        )

    @classmethod
    def create_config(cls, config: str | dict) -> Configuration:
        """Instantiate the Configuration Object from a dictionary or from a path to a config file"""
        config = RunConfiguration.load(config)
        return config

    def patch_checkpoint_lagged_property(self):
        # Patching the self._checkpoint lagged property
        # By default, it assumes forecastor behaviour of retreving n previous steps of data,
        # but we require it to be a list of positive timedeltas from the current date
        # Clear any existing cached value
        if "lagged" in self.checkpoint.__dict__:
            del self.checkpoint.__dict__["lagged"]

        # Monkey patch: replace the property with a simple property that uses our function

        def get_lagged(instance):
            if "lagged" not in instance.__dict__:
                instance.__dict__["lagged"] = checkpoint_lagged_interpolator_patch(instance)
            return instance.__dict__["lagged"]

        # Replace the lagged property on this specific instance
        self.checkpoint.__class__.lagged = property(get_lagged)

    def execute(self) -> None:
        """Execute the interpolator runner with support for multiple interpolation periods."""

        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        lead_time = to_timedelta(self.config.lead_time)
        # This may be used by Output objects to compute the step
        self.lead_time = lead_time
        self.interpolation_window = get_interpolation_window(
            self.checkpoint.data_frequency, self.checkpoint.input_explicit_times
        )
        # Not really timestep but the size of the interpolation window, not sure if this is used
        self.time_step = self.interpolation_window
        input = self.create_input()
        output = self.create_output()

        post_processors = self.post_processors

        # Get the interpolation window size from training config
        boundary_idx = self.checkpoint.input_explicit_times
        # Calculate how many interpolation windows we need for the lead_time
        num_windows = int(lead_time / self.interpolation_window)
        if lead_time % self.interpolation_window != to_timedelta(0):
            LOG.warning(
                f"Lead time {lead_time} is not a multiple of interpolation window {self.interpolation_window}. "
                f"Will interpolate for {num_windows * self.interpolation_window}"
            )

        # Process each interpolation window
        for window_idx in range(num_windows):
            window_start_date = self.config.date + window_idx * self.interpolation_window

            LOG.info(f"Processing interpolation window {window_idx + 1}/{num_windows} starting at {window_start_date}")

            # Create input state for this window
            input_state = input.create_input_state(date=window_start_date)
            self.input_state_hook(input_state)

            # Run interpolation for this window
            for state_idx, state in enumerate(self.run(input_state=input_state, lead_time=self.interpolation_window)):

                # In the first window, we want to write the initial state (t=0)
                # In other windows, we want to skip the initial state (t=0)
                # because it is written as the last state of the previous window
                if window_idx != 0 and state_idx == boundary_idx[0]:
                    continue

                # Updating state step to be a global step not relative to window
                state["step"] = state["step"] + window_idx * self.interpolation_window

                # Apply post-processing
                for processor in post_processors:
                    state = processor.process(state)

                output.write_state(state)

        output.close()

    def predict_step(
        self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, target_forcing: torch.Tensor
    ) -> torch.Tensor:
        return model.predict_step(input_tensor_torch, target_forcing=target_forcing)

    def target_computed_forcings(self, variables: list[str], mask=None) -> list[Forcings]:
        """Create forcings for the bounding target state.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings
        mask : optional
            A mask to apply to the forcings. Defaults to None.

        Returns
        -------
        List[Forcings]
            The created bounding target forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def interpolator_stepper(
        self, start_date: datetime.datetime
    ) -> Generator[tuple[datetime.timedelta, datetime.datetime, int, bool], None, None]:
        """Generate step and date variables for the forecast loop.

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
        target_index : int
            Date used to prepare the next input tensor
        is_last_step : bool
            True if it's the last step of interpolation
        """
        target_steps = self.checkpoint.target_explicit_times
        boundary_idx = self.checkpoint.input_explicit_times
        steps = len(target_steps)

        LOG.info("Time stepping: %s Interpolating %s steps", self.interpolation_window, steps)

        for s in range(steps):
            step = self.interpolation_window * (s + 1) / (boundary_idx[-1] - boundary_idx[0])
            date = start_date + step
            is_last_step = s == steps - 1
            yield step, date, target_steps[s], is_last_step

    def create_target_forcings(
        self, dates: datetime.datetime, state: State, input_tensor_torch: torch.Tensor, interpolation_step: int
    ) -> torch.tensor:
        """Create target forcings tensor.

        Parameters
        ----------
        dates : datetime.datetime
            The dates.
        state : State
            The state dictionary.
        input_tensor_torch : torch.Tensor
            The input tensor.
        interpolation_step : int
            The current interpolation step index
        Returns
        -------
        torch.Tensor
            the target forcings.

        """
        batch_size, _, grid, n_vars = input_tensor_torch.shape

        # Should use self.checkpoint._metadata._config_training.target_forcing.time_fraction but not accessible
        use_time_fraction = True

        # 1 for ensemble size but not needed for the other input??
        target_forcings = torch.empty(
            batch_size,
            1,
            grid,
            len(self.target_forcings) + use_time_fraction,
            device=input_tensor_torch.device,
            dtype=input_tensor_torch.dtype,
        )
        for idx, source in enumerate(self.target_forcings):
            arrays = source.load_forcings_array(dates, state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                target_forcings[..., idx] = torch.tensor(
                    forcing, device=input_tensor_torch.device, dtype=input_tensor_torch.dtype
                )
                self._input_kinds[name] = Kind(forcing=True, constant=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "target forcings")

        if use_time_fraction:
            boundary_times = self.checkpoint.input_explicit_times
            # this only works with two boundary times?
            target_forcings[..., -1] = (interpolation_step - boundary_times[-2]) / (
                boundary_times[-1] - boundary_times[-2]
            )
        return target_forcings

    def forecast(
        self, lead_time: datetime.timedelta, input_tensor_numpy: NDArray, input_state: State
    ) -> Generator[State, None, None]:
        """Interpolate between the current and future state in the input tensor.

        Parameters
        ----------
        lead_time : datetime.timedelta
            Lead time for this interpolation window (should match the window size).
        input_tensor_numpy : NDArray
            The input tensor.
        input_state : State
            The input state. It contains both input dates defined by the config explicit_times.input

        Returns
        -------
        Any
            The forecasted state.
        """
        # This does interpolation but called forecast so we can reuse run()
        self.model.eval()
        torch.set_grad_enabled(False)

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)

        LOG.info("Using autocast %s", self.autocast)

        result = input_state.copy()  # We should not modify the input state
        result["fields"] = dict()
        result["step"] = to_timedelta(0)

        start = input_state["date"]

        reset = np.full((input_tensor_torch.shape[-1],), False)
        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
        typed_variables = self.checkpoint.typed_variables
        for variable, i in variable_to_input_tensor_index.items():
            if typed_variables[variable].is_constant_in_time:
                reset[i] = True

        if self.verbosity > 0:
            self._print_input_tensor("First input tensor", input_tensor_torch)

        # First yield the boundary states (t and t+window_size)
        boundary_times = self.checkpoint.input_explicit_times

        if self.write_initial_state:  # Always True
            # Yield initial boundary state (t)
            initial_result = result.copy()
            initial_result["date"] = start
            initial_result["step"] = to_timedelta(0)
            initial_result["interpolated"] = False
            # Extract fields from the first time step of input tensor
            input_numpy = (
                input_tensor_torch[0, boundary_times[0]].cpu().numpy()
            )  # # Select the initial boundary state (t) - First time step
            for i in range(input_numpy.shape[-1]):
                var_name = None
                for var, idx in variable_to_input_tensor_index.items():
                    if idx == i:
                        var_name = var
                        break
                if var_name and var_name in self.checkpoint.output_tensor_index_to_variable.values():
                    initial_result["fields"][var_name] = input_numpy[:, i]

            yield initial_result

        # Now interpolate between the boundaries
        for s, (step, date, interpolation_step, is_last_step) in enumerate(self.interpolator_stepper(start)):
            title = f"Interpolating step {step}({date})"

            # this should be changed
            result["date"] = date
            result["previous_step"] = result.get("step")
            result["step"] = step
            result["interpolated"] = True

            if self.trace:
                self.trace.write_input_tensor(date, s, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index)

            # Predict next state of atmosphere
            with (
                torch.autocast(device_type=self.device, dtype=self.autocast),
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(title),
            ):
                target_forcing = self.create_target_forcings(date, input_state, input_tensor_torch, interpolation_step)
                y_pred = self.predict_step(self.model, input_tensor_torch, target_forcing=target_forcing)

            # Detach tensor and squeeze (should we detach here?)
            with ProfilingLabel("Sending output to cpu", self.use_profiler):
                output = np.squeeze(y_pred.cpu().numpy())  # shape: (values, variables)

            if self.trace:
                self.trace.write_output_tensor(date, s, output, self.checkpoint.output_tensor_index_to_variable)

            # Update state
            with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                for i in range(output.shape[1]):
                    result["fields"][self.checkpoint.output_tensor_index_to_variable[i]] = output[:, i]

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_output_tensor("Output tensor", output)

            yield result

        # Yield final boundary state (t+window_size) if configured to do so
        if len(boundary_times) > 1:
            final_result = result.copy()
            final_result["date"] = start + self.interpolation_window
            final_result["step"] = self.interpolation_window
            final_result["interpolated"] = False
            # Extract fields from the last time step of input tensor
            if input_tensor_torch.shape[1] > 1:
                input_numpy = input_tensor_torch[0, -1].cpu().numpy()  # Last time step
                for i in range(input_numpy.shape[-1]):
                    var_name = None
                    for var, idx in variable_to_input_tensor_index.items():
                        if idx == i:
                            var_name = var
                            break
                    if var_name and var_name in self.checkpoint.output_tensor_index_to_variable.values():
                        final_result["fields"][var_name] = input_numpy[:, i]
                yield final_result
