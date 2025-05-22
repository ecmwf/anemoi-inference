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
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from numpy.typing import NDArray

from anemoi.inference.runner import Kind
from anemoi.inference.types import State

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..profiler import ProfilingLabel
from . import runner_registry
from .simple import SimpleRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("interpolator")
class InterpolatorRunner(SimpleRunner):
    """A runner to be used for inference of a trained interpolator directly on analysis data
    without being coupled to a forecasting model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the InterpolatorRunner.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.target_forcings = self.target_computed_forcings(
            self.checkpoint._metadata._config_training.target_forcing.data
        )

    def predict_step(
        self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, target_forcing: torch.Tensor
    ) -> torch.Tensor:
        return model.predict_step(input_tensor_torch, target_forcing)

    def target_computed_forcings(self, variables: List[str], mask=None) -> List[Forcings]:
        """Create forcings for the bounding target state.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings

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
    ) -> Generator[Tuple[datetime.timedelta, datetime.datetime, int, bool], None, None]:
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
        target_steps = self.checkpoint._metadata._config_training.explicit_times.target
        steps = len(target_steps)

        LOG.info("%s, time stepping: %s Interpolating %s steps", self.checkpoint.timestep, steps)

        for s in range(steps):
            step = target_steps[s] * self.checkpoint.timestep
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
            boundary_times = self.checkpoint._metadata._config_training.explicit_times.input
            # this only works with two boundary times?
            target_forcings[..., -1] = (interpolation_step - boundary_times[-2]) / (
                boundary_times[-1] - boundary_times[-2]
            )
        return target_forcings

    def forecast(
        self, lead_time: None, input_tensor_numpy: NDArray, input_state: State
    ) -> Generator[State, None, None]:
        """Interpolate between the current and future state in the input tensor.

        Parameters
        ----------
        lead_time : None
            Lead time is not used for interpolation.
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

        for s, (step, date, interpolation_step, is_last_step) in enumerate(self.interpolator_stepper(start)):
            title = f"Interpolating step {step}({date})"

            # this should be changed
            result["date"] = date
            result["previous_step"] = result.get("step")
            result["step"] = self.checkpoint.timestep

            if self.trace:
                self.trace.write_input_tensor(date, s, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index)

            # Predict next state of atmosphere
            with (
                torch.autocast(device_type=self.device, dtype=self.autocast),
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(title),
            ):
                target_forcing = self.create_target_forcings(date, input_state, input_tensor_torch, interpolation_step)
                y_pred = self.predict_step(self.model, input_tensor_torch, target_forcing)

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

            # No need to prepare next input tensor if we are at the last step
            if is_last_step:
                break
