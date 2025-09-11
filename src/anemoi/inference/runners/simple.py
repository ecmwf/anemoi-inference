# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import torch
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer

from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..profiler import ProfilingLabel
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)

# This is because forcings are assumed to already be in the
# state dictionary, so we don't need to load them from anywhere.


class NoForcings(Forcings):
    """No forcings."""

    def __init__(self, context: Any, variables: list[str], mask: IntArray) -> None:
        """Initialize the NoForcings.

        Parameters
        ----------
        context : object
            The context for the forcings.
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.
        """
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(unknown=True)

    def load_forcings_state(self, state: State, date: str) -> None:
        """Load forcings state.

        Parameters
        ----------
        state : State
            The state to load the forcings into.
        date : str
            The date for which to load the forcings.
        """
        pass


@runner_registry.register("simple")
class SimpleRunner(Runner):
    """Use that runner when using the low level API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SimpleRunner.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def create_constant_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create constant computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create dynamic computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created dynamic computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create constant coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created constant coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return []

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create dynamic coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created dynamic coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return []


class SensitivitiesRunner(SimpleRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SimpleRunner.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.perturbed_variable = "2t"

    def perturb_prediction_linearly(
        self, output: torch.Tensor, idx: int, perturbation_perc: float = 0.01
    ) -> torch.Tensor:
        """Perturb the output."""
        # Use a perturbation of 1% of the forecasted value
        pert = torch.zeros_like(output.clone())
        pert[..., idx] = perturbation_perc * output[..., idx]

        return pert

    def predict_step(self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        var_idx = self.checkpoint._metadata.variable_to_output_tensor_index[self.perturbed_variable]

        y_pred = model.predict_step(input_tensor_torch)
        perturbed_output = self.perturb_prediction_linearly(y_pred, idx=var_idx, perturbation_perc=0.01)

        def predict_model_with_torch(x: torch.Tensor) -> torch.Tensor:
            x = x[:, :, None, ...]  # add dummy ensemble dimension as 3rd index
            x = model.pre_processors(x, in_place=False)
            y_hat = model.model(x)
            y_hat = model.post_processors(y_hat, in_place=False)
            return y_hat

        # Compute the sensitivities
        input_tensor_torch.requires_grad_(True)
        with torch.enable_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.autocast):
                y_pred, t_dx_output = torch.autograd.functional.vjp(
                    predict_model_with_torch,
                    input_tensor_torch,
                    v=perturbed_output,
                    create_graph=False,
                    strict=False,
                )
        LOG.info(f"Norm sensitivities (var = {self.perturbed_variable}): {torch.norm(t_dx_output[..., var_idx])}")
        LOG.info(f"Norm input (var = {self.perturbed_variable}): {torch.norm(input_tensor_torch[..., var_idx])}")
        return t_dx_output[0, ...]  # (time, values, variables)

    def forecast(
        self, lead_time: str, input_tensor_numpy: FloatArray, input_state: State
    ) -> Generator[State, None, None]:
        """Forecast the future states.

        Parameters
        ----------
        lead_time : str
            The lead time.
        input_tensor_numpy : FloatArray
            The input tensor.
        input_state : State
            The input state.

        Returns
        -------
        Any
            The forecasted state.
        """
        self.model.eval()

        torch.set_grad_enabled(False)

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)

        lead_time = to_timedelta(lead_time)

        new_state = input_state.copy()  # We should not modify the input state
        new_state["fields"] = dict()
        new_state["step"] = to_timedelta(0)

        start = input_state["date"]

        # The variable `check` is used to keep track of which variables have been updated
        # In the input tensor. `reset` is used to reset `check` to False except
        # when the values are of the constant in time variables

        reset = np.full((input_tensor_torch.shape[-1],), False)
        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
        typed_variables = self.checkpoint.typed_variables
        for variable, i in variable_to_input_tensor_index.items():
            if typed_variables[variable].is_constant_in_time:
                reset[i] = True

        check = reset.copy()

        if self.verbosity > 0:
            self._print_input_tensor("First input tensor", input_tensor_torch)

        for s, (step, date, next_date, is_last_step) in enumerate(self.forecast_stepper(start, lead_time)):
            title = f"Forecasting step {step} ({date})"

            new_state["date"] = date
            new_state["previous_step"] = new_state.get("step")
            new_state["step"] = step

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
                y_pred = self.predict_step(self.model, input_tensor_torch, fcstep=s, step=step, date=date)

            # Update state
            with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                for i in range(t_dx_output.shape[-1]):
                    new_state["fields"][self.checkpoint.input_tensor_index_to_variable[i]] = t_dx_output[:, :, i]

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_input_tensor("Sensitivities tensor", t_dx_output)

            yield new_state
