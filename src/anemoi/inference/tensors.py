# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import sys
from datetime import datetime
from datetime import timedelta
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from anemoi.inference.forcings import BoundaryForcings
from anemoi.inference.forcings import ComputedForcings
from anemoi.inference.forcings import ConstantForcings
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.lazy import torch
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import BoolArray
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


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


class TensorHandler:
    def __init__(
        self,
        runner,
        metadata: Metadata,
        device,
        allow_nans: bool | None,
        constant_forcings_input,
        dynamic_forcings_input,
        boundary_forcings_input,
    ) -> None:
        self.runner = runner
        self.metadata = metadata
        self.device = device
        self.allow_nans = allow_nans

        self.constant_forcings_input = constant_forcings_input
        self.dynamic_forcings_input = dynamic_forcings_input
        self.boundary_forcings_input = boundary_forcings_input

        self.trace = None
        self._input_kinds = {}
        self._input_tensor_by_name = []

        self._output_kinds = {}
        self._output_tensor_by_name = []

        self.constant_forcings_inputs = self.create_constant_forcings_inputs()
        self.dynamic_forcings_inputs = self.create_dynamic_forcings_inputs()
        self.boundary_forcings_inputs = self.create_boundary_forcings_inputs()

        LOG.info("Constant forcings inputs:")
        for f in self.constant_forcings_inputs:
            LOG.info(f"  {f}")

        LOG.info("Dynamic forcings inputs:")
        for f in self.dynamic_forcings_inputs:
            LOG.info(f"  {f}")

        LOG.info("Boundary forcings inputs:")
        for f in self.boundary_forcings_inputs:
            LOG.info(f"  {f}")
        LOG.info("-" * 80)

    def __repr__(self):
        return f"TensorHandler(name={self.name})"

    @property
    def name(self) -> str:
        if self.metadata.multi_dataset:
            return self.metadata.name
        return "data"

    @cached_property
    def lagged(self) -> list[timedelta]:
        """Return the list of steps for the `multi_step_input` fields."""
        result = list(range(0, self.metadata.multi_step_input))

        result = [-s * self.metadata.timestep for s in result]
        return sorted(result)

    def prepare_input_tensor(self, input_state: State, dtype: DTypeLike = np.float32) -> FloatArray:
        """Prepare the input tensor.

        Parameters
        ----------
        input_state : State
            The input state.
        dtype : type, optional
            The data type, by default np.float32.

        Returns
        -------
        FloatArray
            The prepared input tensor.
        """
        if "latitudes" not in input_state:
            input_state["latitudes"] = self.metadata.latitudes

        if "longitudes" not in input_state:
            input_state["longitudes"] = self.metadata.longitudes

        if input_state.get("latitudes") is None or input_state.get("longitudes") is None:
            raise ValueError("Input state must contain 'latitudes' and 'longitudes'")

        typed_variables = self.metadata.typed_variables

        for name in input_state["fields"]:
            self._input_kinds[name] = Kind(input=True, constant=typed_variables[name].is_constant_in_time)

        # Add initial forcings to input state if needed
        self.add_initial_forcings_to_input_state(input_state)

        input_state = self.validate_input_state(input_state)

        input_fields: dict = input_state["fields"]

        input_tensor_numpy = np.full(
            shape=(
                self.metadata.multi_step_input,
                self.metadata.number_of_input_features,
                input_state["latitudes"].size,
            ),
            fill_value=np.nan,
            dtype=dtype,
        )

        self._input_tensor_by_name = [None] * self.metadata.number_of_input_features

        LOG.info("Preparing input tensor with shape %s", input_tensor_numpy.shape)

        variable_to_input_tensor_index = self.metadata.variable_to_input_tensor_index

        check = set()
        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]
            if i in check:
                raise ValueError(f"Duplicate variable {var}/{i} in input fields")
            input_tensor_numpy[:, i] = field
            check.add(i)

            self._input_tensor_by_name[i] = var

        if len(check) != self.metadata.number_of_input_features:
            missing = set(range(self.metadata.number_of_input_features)) - check
            mapping = {v: k for k, v in self.metadata.variable_to_input_tensor_index.items()}
            raise ValueError(f"Missing variables in input fields: {[mapping.get(_, _) for _ in missing]}")

        return input_tensor_numpy

    def validate_input_state(self, input_state: State) -> State:
        """Check that the input state has all expected entries, shapes, and check nans."""

        if not isinstance(input_state, dict):
            raise ValueError("Input state must be a dictionnary")

        EXPECT = dict(date=datetime, latitudes=np.ndarray, longitudes=np.ndarray, fields=dict)

        for key, klass in EXPECT.items():
            if key not in input_state:
                raise ValueError(f"Input state must contain a `{key}` entry")

            if not isinstance(input_state[key], klass):
                raise ValueError(
                    f"Input state entry `{key}` is type {type(input_state[key])}, expected {klass} instead"
                )

        # Detach from the user's input so we can modify it
        input_state = input_state.copy()
        fields = input_state["fields"] = input_state["fields"].copy()
        number_of_grid_points = self.metadata.number_of_grid_points

        for latlon in ("latitudes", "longitudes"):
            if len(input_state[latlon].shape) != 1:
                raise ValueError(f"Input state entry `{latlon}` must be 1D, shape is {input_state[latlon].shape}")

        nlat = len(input_state["latitudes"])
        nlon = len(input_state["longitudes"])
        if nlat != nlon:
            raise ValueError(f"Size mismatch latitudes={nlat}, longitudes={nlon}")

        if nlat != number_of_grid_points:
            raise ValueError(f"Size mismatch latitudes={nlat}, number_of_grid_points={number_of_grid_points}")

        multi_step = self.metadata.multi_step_input

        expected_shape = (multi_step, number_of_grid_points)

        LOG.info("Expected shape for each input fields: %s", expected_shape)

        # Check field
        with_nans = []

        for name, field in list(fields.items()):
            # Allow for 1D fields if multi_step is 1
            if len(field.shape) == 1:
                field = fields[name] = field.reshape(1, field.shape[0])

            if field.shape != expected_shape:
                raise ValueError(f"Field `{name}` has the wrong shape. Expected {expected_shape}, got {field.shape}")

            if np.isinf(field).any():
                raise ValueError(f"Field `{name}` contains infinities")

            if np.isnan(field).any():
                with_nans.append(name)

        if with_nans:
            msg = f"NaNs found in the following variables: {sorted(with_nans)}"
            if self.allow_nans is None:
                LOG.warning(msg)
                self.allow_nans = True

            if not self.allow_nans:
                raise ValueError(msg)

        return input_state

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

        dates = [date + h for h in self.lagged]

        # TODO: Check for user provided forcings
        initial_constant_forcings_inputs = self.runner.initial_constant_forcings_inputs(self.constant_forcings_inputs)
        initial_dynamic_forcings_inputs = self.runner.initial_dynamic_forcings_inputs(self.dynamic_forcings_inputs)

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
            arrays = source.load_forcings_array(dates, input_state)
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

    def create_constant_forcings_inputs(self) -> list[Forcings]:

        result = []

        loaded_variables, loaded_variables_mask = self.metadata.select_variables_and_masks(
            include=["constant+forcing"], exclude=["computed"]
        )

        if len(loaded_variables_mask) > 0:
            result.extend(
                self.create_constant_coupled_forcings(
                    loaded_variables,
                    loaded_variables_mask,
                )
            )

        computed_variables, computed_variables_mask = self.metadata.select_variables_and_masks(
            include=["computed+constant"]
        )

        if len(computed_variables_mask) > 0:
            result.extend(
                self.create_constant_computed_forcings(
                    computed_variables,
                    computed_variables_mask,
                )
            )

        return result

    def create_dynamic_forcings_inputs(self) -> list[Forcings]:

        result = []
        loaded_variables, loaded_variables_mask = self.metadata.select_variables_and_masks(
            include=["forcing"], exclude=["computed", "constant"]
        )

        if len(loaded_variables_mask) > 0:
            result.extend(
                self.create_dynamic_coupled_forcings(
                    loaded_variables,
                    loaded_variables_mask,
                )
            )

        computed_variables, computed_variables_mask = self.metadata.select_variables_and_masks(
            include=["computed"],
            exclude=["constant"],
        )
        if len(computed_variables_mask) > 0:
            result.extend(
                self.create_dynamic_computed_forcings(
                    computed_variables,
                    computed_variables_mask,
                )
            )
        return result

    def create_boundary_forcings_inputs(self) -> list[Forcings]:

        if not self.metadata.has_supporting_array("output_mask"):
            return []

        result = []
        loaded_variables, loaded_variables_mask = self.metadata.select_variables_and_masks(include=["prognostic"])

        if len(loaded_variables_mask) > 0:
            result.extend(
                self.create_boundary_forcings(
                    loaded_variables,
                    loaded_variables_mask,
                )
            )

        return result

    def copy_prognostic_fields_to_input_tensor(
        self, input_tensor_torch: "torch.Tensor", y_pred: "torch.Tensor", check: BoolArray
    ) -> "torch.Tensor":
        """Copy prognostic fields to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        y_pred : torch.Tensor
            The predicted tensor.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        pmask_in = torch.as_tensor(
            self.metadata.prognostic_input_mask,
            device=input_tensor_torch.device,
            dtype=torch.long,
        )

        pmask_out = torch.as_tensor(
            self.metadata.prognostic_output_mask,
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

    def add_dynamic_forcings_to_input_tensor(
        self, input_tensor_torch: "torch.Tensor", state: State, date: datetime, check: BoolArray
    ) -> "torch.Tensor":
        """Add dynamic forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
            The updated input tensor.
        """

        # if self.hacks:
        #     if "dynamic_forcings_date" in self.development_hacks:
        #         date = self.development_hacks["dynamic_forcings_date"]
        #         warnings.warn(f"🧑‍💻 Using `dynamic_forcings_date` hack: {date} 🧑‍💻")

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
        self, input_tensor_torch: "torch.Tensor", state: State, date: datetime, check: BoolArray
    ) -> "torch.Tensor":
        """Add boundary forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
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

    def _print_input_tensor(self, title: str, input_tensors_torch: dict[str, "torch.Tensor"]) -> None:
        """Print the input tensor.

        Parameters
        ----------
        title : str
            The title.
        input_tensors_torch : dict[str, torch.Tensor]
            The input tensors.
        """
        for name, input_tensor_torch in input_tensors_torch.items():
            input_tensor_numpy = input_tensor_torch.cpu().numpy()  # (batch, multi_step_input, values, variables)

            assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
            assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

            input_tensor_numpy = np.squeeze(input_tensor_numpy, axis=0)  # Drop the batch dimension
            input_tensor_numpy = np.swapaxes(input_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

            self._print_tensor(
                f"{title} - dataset: `{name}`", input_tensor_numpy, self._input_tensor_by_name, self._input_kinds
            )

    def _print_output_tensor(self, title: str, output_tensor_numpy: FloatArray) -> None:
        """Print the output tensor.

        Parameters
        ----------
        title : str
            The title.
        output_tensor_numpy : FloatArray
            The output tensor.
        """
        LOG.info(
            "%s",
            f"Output tensor shape={output_tensor_numpy.shape}, NaNs={np.isnan(output_tensor_numpy).sum() / output_tensor_numpy.size: .0%}",
        )

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[1]):
                self._output_tensor_by_name.append(self.metadata.output_tensor_index_to_variable[i])
                if i in self.metadata.prognostic_output_mask:
                    self._output_kinds[self.metadata.output_tensor_index_to_variable[i]] = Kind(prognostic=True)
                else:
                    self._output_kinds[self.metadata.output_tensor_index_to_variable[i]] = Kind(diagnostic=True)

        # output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        if len(output_tensor_numpy.shape) == 2:
            output_tensor_numpy = output_tensor_numpy[np.newaxis, ...]  # Add multi_step_input

        output_tensor_numpy = np.swapaxes(output_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds)

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
        assert tensor_numpy.shape[0] in (1, self.metadata.multi_step_input), tensor_numpy.shape
        assert tensor_numpy.shape[1] == len(tensor_by_name), tensor_numpy.shape
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        console = Console(file=sys.stderr)
        table.add_column("Index", justify="right")
        table.add_column("Variable", justify="left")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("NaNs", justify="center")
        table.add_column("Kind", justify="left")

        for k, v in enumerate(tensor_by_name):
            data = tensor_numpy[-1, k]

            nans = "-"

            if np.isnan(data).any():
                nan_count = np.isnan(data).sum()

                ratio = nan_count / data.size
                nans = f"{ratio:.0%}"

            if np.isinf(data).any():
                nans = "∞"

            table.add_row(
                str(k),
                v,
                f"{np.nanmin(data):g}",
                f"{np.nanmax(data):g}",
                nans,
                str(kinds.get(v, Kind())),
            )

        console.print()
        console.print(table)
        console.print()

    #########################################################################################################
    def create_constant_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = ConstantForcings(self, self.constant_forcings_input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return [result]

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = CoupledForcings(self, self.dynamic_forcings_input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return [result]

    def create_boundary_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        result = BoundaryForcings(self, self.boundary_forcings_input, variables, mask)
        LOG.info("Boundary forcing: %s", result)
        return [result]
