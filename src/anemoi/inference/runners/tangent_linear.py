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
from typing import Any
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from anemoi.utils.text import table

from anemoi.inference.types import BoolArray
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..perturbation import InputPerturbation
from ..profiler import ProfilingLabel
from ..profiler import ProfilingRunner
from .simple import SimpleRunner

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class Kind:
    """Used for debugging purposes."""

    def __init__(self, **kwargs: Any) -> None:
        """Parameters
        -------------
        **kwargs : Any
            Keyword arguments representing the kind attributes.
        """
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """Returns
        ----------
        str
            String representation of the kind.
        """
        result = []

        for k, v in self.kwargs.items():
            if v:
                result.append(k)

        if not result:
            return "?"

        return ", ".join(result)


class TangentLinearRunner(SimpleRunner):
    """Runner that produces the forecast + forward sensitivities ("tangents")."""

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
        self.verbosity = 2

    def wrap_model(self, model: Module) -> Callable:
        """Wrap the model to be used for sensitivities."""

        def model_wrapper(x: Tensor) -> Tensor:
            x = x[:, :, None, ...]  # add dummy ensemble dimension as 3rd index
            x = model.pre_processors(x, in_place=False)
            y_hat = model.model(x)
            y_hat = model.post_processors(y_hat, in_place=False)
            return y_hat

        return model_wrapper

    def predict_step(
        self, model: Module, input_tensor_torch: Tensor, perturbation: Tensor, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        """Primal (forward model) + tangent linear."""
        model_func = self.wrap_model(model)

        LOG.warning(
            "Grad flags: input_tensor_torch %s, perturbation %s",
            input_tensor_torch.requires_grad,
            perturbation.requires_grad
        )

        with torch.autocast(device_type=self.device.type, dtype=self.autocast):
            # NB: torch.autograd.functional.jvp computes the Jacobian-vector product through a "double-backward"
            # https://j-towns.github.io/2017/06/12/A-new-trick.html
            # "true" forward-mode autodiff (torch.func.jvp) requires explicit jvp implementations for things like
            # flash attention and our custom comms autograd functions
            # https://github.com/Dao-AILab/flash-attention/issues/1672
            prediction, perturbation = torch.autograd.functional.jvp(
                model_func,
                input_tensor_torch,
                perturbation,
                create_graph=False,
                strict=False,
            )
                
        return prediction, perturbation

    def forecast(
        self, lead_time: str, input_tensor_numpy: FloatArray, input_state: State, perturbation: InputPerturbation
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

        torch.set_grad_enabled(True)  # we need to propagate gradients forward between forecast steps 

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)
        input_tensor_torch.requires_grad_(True)

        lead_time = to_timedelta(lead_time)

        new_state = input_state.copy()  # We should not modify the input state
        new_state["fields"] = dict()
        new_state["jvp"] = dict()
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

        perturbation = perturbation.create(self.model).to(self.device)
        LOG.warning("Initial perturbation shape: %s", list(perturbation.shape))
        input_perturbation_tensor_torch = perturbation  # .clone()

        with torch.enable_grad():
            for s, (step, date, next_date, is_last_step) in enumerate(self.forecast_stepper(start, lead_time)):
                title = f"Forecasting step {step} ({date})"

                new_state["date"] = date
                new_state["previous_step"] = new_state.get("step")
                new_state["step"] = step

                if self.trace:
                    self.trace.write_input_tensor(
                        date, s, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index, self.checkpoint.timestep
                    )
                    self.trace.write_input_tensor(
                        date,
                        s,
                        input_perturbation_tensor_torch.cpu().numpy(),
                        variable_to_input_tensor_index,
                        self.checkpoint.timestep,
                    )

                # Predict next state of atmosphere
                with (
                    torch.autocast(device_type=self.device.type, dtype=self.autocast),
                    ProfilingLabel("Predict step", self.use_profiler),
                    Timer(title),
                ):
                    y_pred, perturbation = self.predict_step(
                        self.model, input_tensor_torch, perturbation=input_perturbation_tensor_torch, fcstep=s, step=step, date=date
                    )

                assert y_pred.shape == perturbation.shape, f"Shape mismatch: y_pred shape {y_pred.shape} != perturbation shape {perturbation.shape}"

                output = torch.squeeze(y_pred, dim=0)  # shape: (1, values, variables)
                output_perturbation = torch.squeeze(perturbation, dim=0)  # shape: (1, values, variables)

                # Update state
                with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                    for i in range(y_pred.shape[-1]):
                        # y_pred and perturbation have the same shape
                        new_state["fields"][self.checkpoint.output_tensor_index_to_variable[i]] = output[..., i]
                        new_state["jvp"][self.checkpoint.output_tensor_index_to_variable[i]] = output_perturbation[..., i]

                # Prepare input tensor for next step
                if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                    self._print_output_tensor("Output tensor", output.cpu().numpy())
                    self._print_output_tensor("Tangent linear tensor", output_perturbation.cpu().numpy())

                yield new_state

                # No need to prepare next input tensor if we are at the last step
                if is_last_step:
                    break

                # Update  tensor for next iteration
                with ProfilingLabel("Update input tensor for next step", self.use_profiler):
                    check[:] = reset
                    if self.trace:
                        self.trace.reset_sources(reset, self.checkpoint.variable_to_input_tensor_index)

                    input_tensor_torch = self.copy_prognostic_fields_to_input_tensor(input_tensor_torch, y_pred, check)

                    del y_pred  # Recover memory

                    input_tensor_torch = self.add_dynamic_forcings_to_input_tensor(
                        input_tensor_torch, new_state, next_date, check
                    )
                    input_tensor_torch = self.add_boundary_forcings_to_input_tensor(
                        input_tensor_torch, new_state, next_date, check
                    )

                    if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                        self._print_input_tensor("Next input tensor", input_tensor_torch)

                # Update perturbation for next iteration
                with ProfilingLabel("Update tangent linear for next step", self.use_profiler):
                    check[:] = reset
                    if self.trace:
                        self.trace.reset_sources(reset, self.checkpoint.variable_to_input_tensor_index)

                    input_perturbation_tensor_torch = self.copy_prognostic_fields_to_input_tensor(
                        input_perturbation_tensor_torch, perturbation, check
                    )

                    del perturbation  # Recover memory

                    input_perturbation_tensor_torch = self.add_zero_dynamic_forcings_to_input_tensor(
                        input_perturbation_tensor_torch, new_state, next_date, check
                    )
                    input_perturbation_tensor_torch = self.add_zero_boundary_forcings_to_input_tensor(
                        input_perturbation_tensor_torch, new_state, next_date, check
                    )

                    if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                        self._print_input_tensor("Next perturbation tensor", input_perturbation_tensor_torch)

    def copy_prognostic_fields_to_input_tensor(
        self, input_tensor_torch: "Tensor", y_pred: "Tensor", check: BoolArray
    ) -> "Tensor":
        """Copy prognostic fields to the input tensor.

        Parameters
        ----------
        input_tensor_torch : Tensor
            The input tensor.
        y_pred : Tensor
            The predicted tensor.
        check : BoolArray
            The check array.

        Returns
        -------
        Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        pmask_in = torch.as_tensor(
            self.checkpoint.prognostic_input_mask,
            device=input_tensor_torch.device,
            dtype=torch.long,
        )

        pmask_out = torch.as_tensor(
            self.checkpoint.prognostic_output_mask,
            device=y_pred.device,
            dtype=torch.long,
        )  # index_select requires long dtype, can be bool (mask)
        # or int (index) tensors

        prognostic_fields = torch.index_select(y_pred, dim=-1, index=pmask_out)

        input_tensor_torch = input_tensor_torch.roll(-1, dims=1)
        input_tensor_torch[:, -1, :, pmask_in] = prognostic_fields

        pmask_in_np = pmask_in.detach().cpu().numpy()
        if check[pmask_in_np].any():
            # Report which ones are conflicting
            conflicting = [self._input_tensor_by_name[i] for i in pmask_in_np[check[pmask_in_np]]]
            raise AssertionError(
                f"Attempting to overwrite existing prognostic input slots for variables: {conflicting}"
            )

        check[pmask_in_np] = True

        for n in pmask_in_np:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)
            if self.trace:
                self.trace.from_rollout(self._input_tensor_by_name[n])

        return input_tensor_torch

    def run(
        self,
        *,
        input_state: State,
        perturbation: InputPerturbation,
        lead_time: str | int | datetime.timedelta,
        return_numpy: bool = True,
    ) -> Generator[State, None, None]:
        """Run the model.

        Parameters
        ----------
        input_state : State
            The input state.
        lead_time : Union[str, int, datetime.timedelta]
            The lead time.
        return_numpy : bool, optional
            Whether to return the output state fields as numpy arrays, by default True.
            Otherwise, it will return torch tensors.

        Returns
        -------
        Generator[State, None, None]
            The forecasted state.
        """
        # Shallow copy to avoid modifying the user's input state
        input_state = input_state.copy()
        input_state["fields"] = input_state["fields"].copy()

        self.constant_forcings_inputs = self.create_constant_forcings_inputs(input_state)
        self.dynamic_forcings_inputs = self.create_dynamic_forcings_inputs(input_state)
        self.boundary_forcings_inputs = self.create_boundary_forcings_inputs(input_state)

        LOG.warning("-" * 80)
        LOG.warning("Input state:")
        LOG.warning(f"  {list(input_state['fields'].keys())}")

        LOG.warning("Constant forcings inputs:")
        for f in self.constant_forcings_inputs:
            LOG.warning(f"  {f}")

        LOG.warning("Dynamic forcings inputs:")
        for f in self.dynamic_forcings_inputs:
            LOG.warning(f"  {f}")

        LOG.warning("Boundary forcings inputs:")
        for f in self.boundary_forcings_inputs:
            LOG.warning(f"  {f}")
        LOG.warning("-" * 80)

        lead_time = to_timedelta(lead_time)

        with ProfilingRunner(self.use_profiler):
            with ProfilingLabel("Prepare input tensor", self.use_profiler):
                input_tensor = self.prepare_input_tensor(input_state)

            try:
                yield from self.prepare_output_state(
                    self.forecast(lead_time, input_tensor, input_state, perturbation), return_numpy
                )
            except (TypeError, ModuleNotFoundError, AttributeError):
                if self.report_error:
                    self.checkpoint.report_error()
                raise

    def add_dynamic_forcings_to_input_tensor(
        self, input_tensor_torch: "Tensor", state: State, date: datetime.datetime, check: BoolArray
    ) -> "Tensor":
        """Add dynamic forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        Tensor
            The updated input tensor.
        """

        if self.hacks:
            if "dynamic_forcings_date" in self.development_hacks:
                date = self.development_hacks["dynamic_forcings_date"]
                import warnings
                warnings.warn(f"🧑‍💻 Using `dynamic_forcings_date` hack: {date} 🧑‍💻")

        # TODO: check if there were not already loaded as part of the input state

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        for source in self.dynamic_forcings_inputs:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)

            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device

            input_tensor_torch[:, -1, :, source.mask] = forcings  # Copy forcings to last 'multi_step_input' row

            assert not check[source.mask].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(forcing=True, **source.kinds)

            if self.trace:
                for n in source.mask:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "dynamic forcings")

        return input_tensor_torch

    def add_boundary_forcings_to_input_tensor(
        self, input_tensor_torch: "Tensor", state: State, date: datetime.datetime, check: BoolArray
    ) -> "Tensor":
        """Add boundary forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        sources = self.boundary_forcings_inputs
        for source in sources:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device
            total_mask = np.ix_([0], [-1], source.spatial_mask, source.variables_mask)
            input_tensor_torch[total_mask] = forcings  # Copy forcings to last 'multi_step_input' row

            for n in source.variables_mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(boundary=True, forcing=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "boundary forcings")

        # TO DO: add some consistency checks as above
        return input_tensor_torch

    def add_zero_dynamic_forcings_to_input_tensor(
        self, input_tensor_torch: "Tensor", state: State, date: datetime.datetime, check: BoolArray
    ) -> "Tensor":
        """Add all-zero (dummy) dynamic forcings to an input-like tensor.

        Parameters
        ----------
        input_tensor_torch : Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        Tensor
            The updated input tensor.
        """

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        for source in self.dynamic_forcings_inputs:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)
            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension
            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device

            input_tensor_torch[:, -1, :, source.mask] = torch.zeros_like(forcings)  # Copy forcings to last 'multi_step_input' row

            assert not check[source.mask].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(forcing=True, **source.kinds)

            if self.trace:
                for n in source.mask:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "dynamic forcings")

        return input_tensor_torch

    def add_zero_boundary_forcings_to_input_tensor(
        self, input_tensor_torch: "Tensor", state: State, date: datetime.datetime, check: BoolArray
    ) -> "Tensor":
        """Add boundary forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        sources = self.boundary_forcings_inputs
        for source in sources:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)
            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension
            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device
            total_mask = np.ix_([0], [-1], source.spatial_mask, source.variables_mask)
            
            input_tensor_torch[total_mask] = torch.zeros_like(forcings)  # Copy all-zero forcings to last 'multi_step_input' row

            for n in source.variables_mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(boundary=True, forcing=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "boundary forcings")

        # TO DO: add some consistency checks as above
        return input_tensor_torch

    def _print_tensor(
        self, title: str, tensor_numpy: FloatArray, tensor_by_name: list[str], kinds: dict[str, Kind]
    ) -> None:
        """Print the tensor.

        Parameters
        ----------
        title : str
            The title.
        tensor_numpy : FloatArray
            The tensor.
        tensor_by_name : list
            The tensor by name.
        kinds : dict
            The kinds.
        """
        assert len(tensor_numpy.shape) == 3, tensor_numpy.shape
        assert tensor_numpy.shape[0] in (1, self.checkpoint.multi_step_input), tensor_numpy.shape
        assert tensor_numpy.shape[1] == len(tensor_by_name), tensor_numpy.shape

        t = []
        for k, v in enumerate(tensor_by_name):
            data = tensor_numpy[-1, k]

            nans = "-"

            if np.isnan(data).any():
                nan_count = np.isnan(data).sum()

                ratio = nan_count / data.size
                nans = f"{ratio:.0%}"

            if np.isinf(data).any():
                nans = "∞"

            t.append((k, v, np.nanmin(data), np.nanmax(data), nans, kinds.get(v, Kind())))

        LOG.warning("")
        LOG.warning(
            "%s:\n\n%s\n", title, table(t, header=["Index", "Variable", "Min", "Max", "NaNs", "Kind"], align="><<<|<")
        )
        LOG.warning("")

    def _print_input_tensor(self, title: str, input_tensor_torch: "Tensor") -> None:
        """Print the input tensor.

        Parameters
        ----------
        title : str
            The title.
        input_tensor_torch : Tensor
            The input tensor.
        """
        input_tensor_numpy = input_tensor_torch.clone().detach().cpu().numpy()  # (batch, multi_step_input, values, variables)

        assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
        assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

        input_tensor_numpy = np.squeeze(input_tensor_numpy, axis=0)  # Drop the batch dimension
        input_tensor_numpy = np.swapaxes(input_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(title, input_tensor_numpy, self._input_tensor_by_name, self._input_kinds)

    def _print_output_tensor(self, title: str, output_tensor_numpy: FloatArray) -> None:
        """Print the output tensor.

        Parameters
        ----------
        title : str
            The title.
        output_tensor_numpy : FloatArray
            The output tensor.
        """
        LOG.warning(
            "%s",
            f"Output tensor shape={output_tensor_numpy.shape}, NaNs={np.isnan(output_tensor_numpy).sum() / output_tensor_numpy.size: .0%}",
        )

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[-1]):
                self._output_tensor_by_name.append(self.checkpoint.output_tensor_index_to_variable[i])
                if i in self.checkpoint.prognostic_output_mask:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(prognostic=True)
                else:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(diagnostic=True)

        # output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        if len(output_tensor_numpy.shape) == 2:
            output_tensor_numpy = output_tensor_numpy[np.newaxis, ...]  # Add multi_step_input

        output_tensor_numpy = np.swapaxes(output_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds)
