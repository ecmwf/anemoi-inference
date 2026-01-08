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
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from numpy.typing import NDArray

from anemoi.inference.config import Configuration
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.device import get_available_device
from anemoi.inference.lazy import torch
from anemoi.inference.runner import Kind
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..forcings import ComputedInterpForcings
from ..forcings import ConstantInterpForcings
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
        self.from_analysis = any("use_original_paths" in keys for keys in config.input.values())
        self.device = get_available_device()
        self.patch_checkpoint_lagged_property()
        assert (
            self.config.write_initial_state
        ), "Interpolator output should include temporal start state, end state and boundary conditions"

        if hasattr(self.checkpoint._metadata._config_training, "target_forcing"):
            self.target_forcings = self.target_computed_forcings(
                self.checkpoint._metadata._config_training.target_forcing.data
            )

        # This may be used by Output objects to compute the step
        self.interpolation_window = get_interpolation_window(
            self.checkpoint.data_frequency, self.checkpoint.input_explicit_times
        )

        self.multi_step_input = 2
        self.constants_input = None

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
        # This may be used by Output objects to compute the step
        self.interpolation_window = get_interpolation_window(
            self.checkpoint.data_frequency, self.checkpoint.input_explicit_times
        )
        self.lead_time = to_timedelta(self.config.lead_time)
        self.time_step = self.interpolation_window

        self.lead_time = to_timedelta(self.config.lead_time)
        self.time_step = self.interpolation_window

    @classmethod
    def create_config(cls, config: str | dict) -> Configuration:
        """Instantiate the Configuration Object from a dictionary or from a path to a config file"""
        config = RunConfiguration.load(config)
        return config

    def patch_data_request(self, request: dict) -> dict:
        """Set sensible defaults when this runner is used with the `retrieve` command."""
        req = request.copy()
        req["class"] = "od"
        req["type"] = "fc"
        req["stream"] = "oper"

        # by default the `time` will be two initialisation times, e.g. 0000 and 0600
        # instead, we want one initialisation time and use `step` to get the input forecast based on the lead time.
        # set time based on the date in the config (reference_date), or already set by the `retrieve` command
        if self.reference_date:
            req["time"] = f"{self.reference_date.hour*100:04d}"
        else:
            req["time"] = sorted(req.get("time", ["0000"]))[0]
        req["step"] = (
            f"0/to/{int(self.lead_time.total_seconds()//3600)}/by/{int(self.interpolation_window.total_seconds()//3600)}"
        )
        return super().patch_data_request(req)

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

    def create_input_state(self, *, date: datetime.datetime, **kwargs) -> State:
        prognostic_input = self.create_prognostics_input()
        LOG.info("📥 Prognostic input: %s", prognostic_input)
        prognostic_state = prognostic_input.create_input_state(
            date=date, start_date=self.config.date, ref_date_index=0, **kwargs
        )
        self._check_state(prognostic_state, "prognostics")

        forcings_input = self.create_dynamic_forcings_input()
        LOG.info("📥 Dynamic forcings input: %s", forcings_input)
        forcings_state = forcings_input.create_input_state(
            date=date, start_date=self.config.date, ref_date_index=0, **kwargs
        )
        self._check_state(forcings_state, "dynamic_forcings")

        self.constants_state["date"] = prognostic_state["date"]

        input_state = self._combine_states(
            prognostic_state,
            self.constants_state,
            forcings_state,
        )

        return input_state

    def execute(self) -> None:
        """Execute the interpolator runner with support for multiple interpolation periods."""
        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        output = self.create_output()

        post_processors = self.post_processors

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

        if self.constants_input is None:
            self.constants_input = self.create_constant_coupled_forcings_input()
            LOG.info("📥 Constant forcings input: %s", self.constants_input)
            self.constants_state = self.constants_input.create_input_state(
                date=self.config.date, constant=True, ref_date_index=0
            )
            for key in self.constants_state["fields"].keys():
                self.constants_state["fields"][key] = np.concatenate(
                    [self.constants_state["fields"][key], self.constants_state["fields"][key]], axis=0
                )
            self._check_state(self.constants_state, "constant_forcings")

        # Process each interpolation window
        for window_idx in range(num_windows):
            window_start_date = self.config.date + window_idx * self.interpolation_window

            LOG.info(
                "Processing interpolation window %d/%d starting at %s", window_idx + 1, num_windows, window_start_date
            )

            input_state = self.create_input_state(date=window_start_date)

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
        self, model: "torch.nn.Module", input_tensor_torch: "torch.Tensor", target_forcing: "torch.Tensor"
    ) -> "torch.Tensor":
        return model.predict_step(input_tensor_torch, target_forcing=target_forcing)

    def add_initial_forcings_to_input_state(self, input_state: State) -> None:
        """Add initial forcings to the input state.

        Parameters
        ----------
        input_state : State
            The input state.
        """
        # Should that be alreay a list of dates
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.checkpoint.lagged]

        # For output object. Should be moved elsewhere
        self.reference_date = self.reference_date or date
        self.reference_dates = [self.reference_date + h for h in self.checkpoint.lagged]
        self.initial_dates = dates

        # TODO: Check for user provided forcings

        # We may need different forcings initial conditions
        initial_constant_forcings_inputs = self.initial_constant_forcings_inputs(self.constant_forcings_inputs)
        initial_dynamic_forcings_inputs = self.initial_dynamic_forcings_inputs(self.dynamic_forcings_inputs)

        LOG.info("-" * 80)
        LOG.info("Initial forcings:")
        LOG.info("  Constant forcings inputs:")
        for f in initial_constant_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("  Dynamic forcings inputs:")
        for f in initial_dynamic_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("-" * 80)

        for source in initial_constant_forcings_inputs:
            LOG.info("Constant forcings input: %s %s (%s)", source, source.variables, dates)
            arrays = source.load_forcings_array(self.reference_dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial constant forcings")

        for source in initial_dynamic_forcings_inputs:
            LOG.info("Dynamic forcings input: %s %s (%s)", source, source.variables, dates)
            arrays = source.load_forcings_array(dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=False, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial dynamic forcings")

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
        result = ComputedInterpForcings(self, variables, mask)
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

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create constant coupled forcings.

        Parameters
        ----------
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant coupled forcings.
        """
        input = self.create_constant_coupled_forcings_input()
        result = ConstantInterpForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)

        return [result]

    def create_target_forcings(
        self, dates: datetime.datetime, state: State, input_tensor_torch: "torch.Tensor", interpolation_step: int
    ) -> "torch.tensor":
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
                self.trace.write_input_tensor(
                    date, s, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index, self.checkpoint.timestep
                )

            # Predict next state of atmosphere
            with (
                torch.autocast(device_type=self.device.type, dtype=self.autocast),
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(title),
            ):
                target_forcing = self.create_target_forcings(date, input_state, input_tensor_torch, interpolation_step)
                y_pred = self.predict_step(self.model, input_tensor_torch, target_forcing=target_forcing)

            # Detach tensor and squeeze (should we detach here?)
            with ProfilingLabel("Sending output to cpu", self.use_profiler):
                output = np.squeeze(y_pred.cpu().numpy())  # shape: (values, variables)

            if self.trace:
                self.trace.write_output_tensor(
                    date, s, output, self.checkpoint.output_tensor_index_to_variable, self.checkpoint.timestep
                )

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
