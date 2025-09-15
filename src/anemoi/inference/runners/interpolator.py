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



def checkpoint_lagged_interpolator_patch(self) -> list[datetime.timedelta]:
    # For interpolator, we always want positive timedeltas
    result = [s * to_timedelta(self.checkpoint.timestep) for s in self.input_explicit_times]
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
        self.time_step = self.checkpoint.timestep
        self.interpolation_window = to_timedelta(self.checkpoint.timestep) * (input_explicit_times[1] - input_explicit_times[0])
        # Not really timestep but the size of the interpolation window, not sure if this is used
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
        input_tensor_torch = torch.from_numpy(input_tensor_numpy).to(self.device) # (bs, ts, ens, grid, n_vars)

        LOG.info("Using autocast %s", self.autocast)

        template_result = input_state.copy()  # We should not modify the input state
        template_result["fields"] = dict()
        template_result["step"] = to_timedelta(0)

        start = input_state["date"]

        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
        output_tensor_index_to_variable = self.checkpoint.output_tensor_index_to_variable

        # First yield the boundary states (t and t+window_size)
        boundary_times = self.checkpoint.input_explicit_times

        if self.write_initial_state:  # Always True
            # Yield initial boundary state (t)
            initial_result = template_result.copy()
            initial_result["date"] = start
            initial_result["fields"] = dict()
            initial_result["step"] = to_timedelta(0)
            # Extract fields from the first time step of input tensor
            input_numpy = (
                input_tensor_torch[:, boundary_times.index(0) ].cpu().numpy() # shape( bs, ens, grid, n_vars)
            )  # # Select the initial boundary state (t) - First time step
            for var_name, i in variable_to_input_tensor_index.items():

                if self.model.mass_conserving_accums and var_name in self.model.mass_conserving_accums.values():
                    var_name = {v: k for k, v in self.model.mass_conserving_accums.items()}[var_name]

                # elif var_name not in self.checkpoint.output_tensor_index_to_variable.values():
                #     continue
                
                initial_result["fields"][var_name] = input_numpy[0, 0, :, i] # NOTE: we currently implement such that bs=1 and ens=1, ensemble not supported

            yield initial_result

        steps, dates, target_indices = self.interpolator_stepper(start)
        title = f"Interpolating steps {steps}"

        if self.trace:
            self.trace.write_input_tensor(dates, steps, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index)

        with ( 
            torch.autocast(device_type=self.device, dtype=self.autocast),
            ProfilingLabel("Predict step", self.use_profiler),
            Timer(title),
        ):
            target_forcing = self.create_target_forcings(dates, input_state, input_tensor_torch, target_indices) # shape(bs, interpolation_steps, ens, grid, forcing_dim)

            # The output of self.predict_step is shape(bs, interpolation_steps, ens, grid, n_vars) or shape(bs, interpolation_steps+1, ens, grid, n_vars) depending on whether mass conservation required an adjusted final timestep value away from the initial boundary value
            y_pred = self.predict_step(self.model, input_tensor_torch, target_forcing=target_forcing, include_right_boundary=True)

            with ProfilingLabel("Sending output to cpu", self.use_profiler):
                y_pred_cpu = y_pred.cpu().numpy()  # shape: (bs, interpolation_steps, ens, grid, variables)

            if self.trace:
                self.trace.write_output_tensor(dates, steps, output, self.checkpoint.output_tensor_index_to_variable)

            for tidx, (step, date) in enumerate(zip(steps, dates)):
                result_step = template_result.copy()
                result_step["date"] = date
                result_step["step"] = step
                result_step["interpolated"] = True
                result_step["fields"] = dict()
                for i, var_name in output_tensor_index_to_variable.items():
                    result_step["fields"][var_name] = y_pred_cpu[0, tidx, 0, :, i] #NOTE - currently since bs=1 and ens=1 
                yield result_step

        # Yield final boundary state
        result_right_boundary = input_state.copy()
        result_right_boundary["fields"] = dict()
        result_right_boundary["date"] = start
        result_right_boundary["step"] = self.checkpoint.timestep + steps[-1]
        # Extract fields from the last time step of output tensor

        for idx, var_name in output_tensor_index_to_variable.items():
            result_right_boundary["fields"][var_name] = y_pred_cpu[0, self.checkpoint.input_explicit_times[-1]-1, 0, :, idx]

        yield result_right_boundary

    def predict_step(
        self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, target_forcing: torch.Tensor, include_right_boundary: bool = False
    ) -> torch.Tensor:
        return model.predict_step(input_tensor_torch, target_forcing=target_forcing, include_right_boundary=include_right_boundary)
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
        """
        target_steps = self.checkpoint.target_explicit_times
        boundary_idx = self.checkpoint.input_explicit_times
        steps = len(target_steps)

        LOG.info("Time stepping: %s Interpolating %s steps", self.checkpoint.timestep, steps)

        time_steps = [self.checkpoint.timestep * (s + 1) for s in range(steps)]
        dates = [start_date + time_step for time_step in time_steps]
        target_indices = [target_steps[s] for s in range(steps)]

        return time_steps, dates, target_indices

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

