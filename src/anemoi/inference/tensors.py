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
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from anemoi.inference.forcings import BoundaryForcings
from anemoi.inference.forcings import ComputedForcings
from anemoi.inference.forcings import ConstantForcings
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.lazy import torch
from anemoi.inference.types import BoolArray
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.forcings import Forcings
    from anemoi.inference.input import Input
    from anemoi.inference.metadata import Metadata
    from anemoi.inference.runner import Runner

LOG = logging.getLogger(__name__)


class Kind:
    """Used for debugging purposes."""

    def __init__(self, **attributes: Any):
        self.attributes = attributes

    def __repr__(self) -> str:
        result = []
        for k, v in self.attributes.items():
            if v:
                result.append(k)
        if not result:
            return "?"
        return ", ".join(result)


class TensorHandler:
    """The TensorHandler is responsible for creating the input tensor for one dataset.
    It also handles loading the forcings and copying prognostic variables from the output tensor to the input tensor during rollout.
    A handler should be created per dataset. The metadata and inputs provided to the handler are specific to that dataset.
    """

    def __init__(
        self,
        context: "Runner",
        metadata: "Metadata",
        constant_forcings_input: "Input",
        dynamic_forcings_input: "Input",
        boundary_forcings_input: "Input",
        trace_path: str | None = None,
    ) -> None:
        self.context = context
        self.metadata = metadata

        self.constant_forcings_input = constant_forcings_input
        self.dynamic_forcings_input = dynamic_forcings_input
        self.boundary_forcings_input = boundary_forcings_input

        self._input_kinds = {}
        self._input_tensor_by_name = []
        self._input_units = {}

        self._output_kinds = {}
        self._output_tensor_by_name = []
        self._output_units = {}

        self.constant_forcings_providers = self.create_constant_forcings_providers()
        self.dynamic_forcings_providers = self.create_dynamic_forcings_providers()
        self.boundary_forcings_providers = self.create_boundary_forcings_providers()

        LOG.info("-" * 80)
        LOG.info("Constant forcings providers:")
        for f in self.constant_forcings_providers:
            LOG.info(f"  {f}")

        LOG.info("Dynamic forcings providers:")
        for f in self.dynamic_forcings_providers:
            LOG.info(f"  {f}")

        LOG.info("Boundary forcings providers:")
        for f in self.boundary_forcings_providers:
            LOG.info(f"  {f}")
        LOG.info("-" * 80)

        if trace_path:
            from .trace import Trace

            trace_path = trace_path.format(dataset=self.dataset_name, dataset_name=self.dataset_name)
            self.trace = Trace(path=trace_path)
        else:
            self.trace = None

    def __repr__(self):
        return f"TensorHandler(dataset={self.dataset_name})"

    @property
    def dataset_name(self) -> str:
        """Name of the dataset associated with the tensor handler."""
        return self.metadata.dataset_name

    def prepare_input_tensor(self, input_state: State, dtype: DTypeLike = np.float32) -> FloatArray:
        """Prepare the input tensor from the input state."""
        if "latitudes" not in input_state:
            input_state["latitudes"] = self.metadata.latitudes

        if "longitudes" not in input_state:
            input_state["longitudes"] = self.metadata.longitudes

        if input_state.get("latitudes") is None or input_state.get("longitudes") is None:
            raise ValueError("Input state must contain 'latitudes' and 'longitudes'")

        typed_variables = self.metadata.typed_variables

        for name in input_state["fields"]:
            self._input_kinds[name] = Kind(input=True, constant=typed_variables[name].is_constant_in_time)
            self._input_units[name] = typed_variables[name].units

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

        LOG.info(f"Preparing input tensor with shape {input_tensor_numpy.shape}")

        variable_to_input_tensor_index = self.metadata.variable_to_input_tensor_index

        check = set()
        for var, field in input_fields.items():
            if var not in variable_to_input_tensor_index:
                continue
            i = variable_to_input_tensor_index[var]
            if i in check:
                raise ValueError(f"Duplicate variable {var}/{i} in input fields")
            input_tensor_numpy[:, i] = field
            check.add(i)

            self._input_tensor_by_name[i] = var

        if len(check) != self.metadata.number_of_input_features:
            missing = set(range(self.metadata.number_of_input_features)) - check
            mapping = {v: k for k, v in self.metadata.variable_to_input_tensor_index.items()}
            # Input-tensor variables categorised as diagnostic (e.g. observation
            # channels in obs-forecaster checkpoints) are model inputs only where
            # observations exist. When absent they are left as NaN so the model's
            # imputer applies the no-obs sentinel; runners that assimilate them
            # (e.g. obs_da_cycling) fill the relevant slots later.
            categories = self.metadata.variable_categories()
            optional = {i for i in missing if "diagnostic" in categories.get(mapping.get(i), [])}
            missing = missing - optional
            if missing:
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

        LOG.info(f"Expected shape for each input fields: {expected_shape}")

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
            if self.context.allow_nans is None:
                LOG.warning(msg)
                self.context.allow_nans = True

            if not self.context.allow_nans:
                raise ValueError(msg)

        return input_state

    def add_initial_forcings_to_input_state(self, input_state: State) -> None:
        """Add initial forcings to the input state.

        Parameters
        ----------
        input_state : State
            The input state.
        """
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.metadata.lagged]

        initial_constant_forcings_providers = self.initial_constant_forcings_providers(self.constant_forcings_providers)
        initial_dynamic_forcings_providers = self.initial_dynamic_forcings_providers(self.dynamic_forcings_providers)

        LOG.info("-" * 80)
        LOG.info("Initial forcings providers:")
        LOG.info("  Constant forcings:")
        for f in initial_constant_forcings_providers:
            LOG.info(f"    {f}")
        LOG.info("  Dynamic forcings:")
        for f in initial_dynamic_forcings_providers:
            LOG.info(f"    {f}")
        LOG.info("Initial forcings dates:")
        LOG.info(f"  {', '.join([date.isoformat() for date in dates])}")

        for provider in initial_constant_forcings_providers:
            if all(
                name in fields and isinstance(fields[name], np.ndarray) and fields[name].size > 0
                for name in provider.variables
            ):
                LOG.info(f"Skipping initial constant forcings {provider}, all variables are present in the input state")
                continue
            arrays = provider.load_forcings_array(dates, input_state)
            for name, forcing in zip(provider.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=True, **provider.kinds)
                if self.trace:
                    self.trace.from_source(name, provider, "initial constant forcings")

        for provider in initial_dynamic_forcings_providers:
            if all(
                name in fields and isinstance(fields[name], np.ndarray) and fields[name].size > 0
                for name in provider.variables
            ):
                LOG.info(f"Skipping initial dynamic forcings {provider}, all variables are present in the input state")
                continue
            arrays = provider.load_forcings_array(dates, input_state)
            for name, forcing in zip(provider.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=False, **provider.kinds)
                if self.trace:
                    self.trace.from_source(name, provider, "initial dynamic forcings")

        LOG.info("-" * 80)

    def create_constant_forcings_providers(self) -> list["Forcings"]:
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

    def create_dynamic_forcings_providers(self) -> list["Forcings"]:
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

        # Also create providers for computed variables that are decoder-only forcings
        # (not in the input tensor but needed by load_decoder_forcings)
        df_vars = set(self.metadata.decoder_forcing_variables)
        v2i = self.metadata.variable_to_input_tensor_index
        all_computed = self.metadata.select_variables(include=["computed"], exclude=["constant"], has_mars_requests=False)
        decoder_only_computed = [v for v in all_computed if v in df_vars and v not in v2i]
        if decoder_only_computed:
            result.extend(
                self.create_dynamic_computed_forcings(
                    decoder_only_computed,
                    np.array([], dtype=int),
                )
            )

        # Create coupled providers for decoder-only forcings that are NOT computed
        # (e.g. insolation, gridsat_cos_sza, satellite viewing geometry).
        # These must be loaded from the dataset at each forecast step.
        loaded_set = set(loaded_variables)
        computed_set = set(all_computed) | set(decoder_only_computed)
        decoder_only_loaded = [v for v in self.metadata.decoder_forcing_variables
                               if v not in loaded_set and v not in computed_set and v not in v2i]
        if decoder_only_loaded:
            result.extend(
                self.create_dynamic_coupled_forcings(
                    decoder_only_loaded,
                    np.array([], dtype=int),
                )
            )

        return result

    def create_boundary_forcings_providers(self) -> list["BoundaryForcings"]:
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

    def initial_constant_forcings_providers(self, constant_forcings_providers: list[Forcings]) -> list[Forcings]:
        """Modify the constant forcings providers for the first step."""
        # Give an opportunity to modify the forcings for the first step
        return constant_forcings_providers

    def initial_dynamic_forcings_providers(self, dynamic_forcings_providers: list[Forcings]) -> list[Forcings]:
        """Modify the dynamic forcings providers for the initial step of the inference process.

        This method provides a hook to adjust the list of dynamic forcings before the first
        inference step is executed. By default, it returns the inputs unchanged, but subclasses
        can override this method to implement custom preprocessing or initialization logic.
        """
        # Give an opportunity to modify the forcings for the first step
        return dynamic_forcings_providers

    def copy_prognostic_fields_to_input_tensor(
        self, input_tensor_torch: "torch.Tensor", y_pred: "torch.Tensor", check: BoolArray
    ) -> "torch.Tensor":
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
        keep_steps = min(self.metadata.multi_step_output, self.metadata.multi_step_input)
        input_tensor_torch = input_tensor_torch.roll(-keep_steps, dims=1)

        for i in range(keep_steps):
            input_tensor_torch[:, -(i + 1), :, pmask_in] = prognostic_fields[:, -(i + 1), ...]

        # Corrector slots (satellite viewing geometry, reportype, ...) describe
        # observations, which don't exist for model-advanced states. Training zeroes
        # them when advancing the input (rollout _advance_dataset_input); write NaN
        # here so the checkpoint's imputer applies the same no-obs sentinel. Without
        # this, values written during DA cycling would wrap around via the roll above
        # and persist through the whole forecast. No-op when there are no correctors.
        cmask_in = self.metadata.corrector_input_mask
        if len(cmask_in) > 0:
            cmask = torch.as_tensor(cmask_in, device=input_tensor_torch.device, dtype=torch.long)
            for i in range(keep_steps):
                input_tensor_torch[:, -(i + 1), :, cmask] = torch.nan

        pmask_in_np = pmask_in.detach().cpu().numpy()
        if check[pmask_in_np].any():
            # Report which ones are conflicting
            conflicting = [self._input_tensor_by_name[i] for i in pmask_in_np[check[pmask_in_np]]]
            raise AssertionError(
                f"[{self.dataset_name}] Attempting to overwrite existing prognostic input slots for variables: {conflicting}"
            )

        check[pmask_in_np] = True

        for n in pmask_in_np:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)
            if self.trace:
                self.trace.from_rollout(self._input_tensor_by_name[n])

        return input_tensor_torch

    def add_dynamic_forcings_to_input_tensor(
        self,
        input_tensor_torch: "torch.Tensor",
        state: State,
        dates: list[datetime],
        check: BoolArray,
    ) -> "torch.Tensor":
        # TODO: re-enable
        # if self.hacks:
        #     if "dynamic_forcings_date" in self.development_hacks:
        #         date = self.development_hacks["dynamic_forcings_date"]
        #         dates = [date]
        #         warnings.warn(f"🧑‍💻 Using `dynamic_forcings_date` hack: {date} 🧑‍💻")

        # TODO: check if there were not already loaded as part of the input state

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        for source in self.dynamic_forcings_providers:
            # Skip decoder-only forcings providers (empty mask = no input tensor slot)
            if len(source.mask) == 0:
                continue

            forcings = source.load_forcings_array(dates, state)  # shape: (variables, dates, values)

            forcings = np.swapaxes(forcings, 0, 1)  # shape: (dates, variable, values)

            forcings = np.swapaxes(
                forcings[np.newaxis, :, np.newaxis, ...], -2, -1
            )  # shape: (1, dates, 1, values, variables)

            forcings = torch.from_numpy(forcings).to(self.context.device)  # Copy to device

            for i in range(min(self.metadata.multi_step_output, self.metadata.multi_step_input)):
                input_tensor_torch[:, -(i + 1), :, source.mask] = forcings[
                    :, -(i + 1), ...
                ]  # Copy forcings to corresponding 'multi_step_input' row

            assert not check[source.mask].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(forcing=True, **source.kinds)

            if self.trace:
                for n in source.mask:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "dynamic forcings")

        return input_tensor_torch

    def load_decoder_forcings(
        self,
        state: State,
        dates: list[datetime],
    ) -> "torch.Tensor | None":
        """Load decoder-forcing variables at the target date(s).

        Decoder-forcings are injected only into the decoder and describe
        conditions at the time being predicted (e.g. satellite viewing
        geometry at t+timestep). They are loaded from the dataset's
        dynamic-forcings providers, then stacked in ascending data-space
        position order to match how training extracts them via
        ``batch[..., data_indices.data.input.decoder_forcing]``.

        Returns
        -------
        torch.Tensor | None
            Tensor of shape ``(1, n_step_output, 1, grid, n_decoder_forcing)``
            in ascending data-position order, or ``None`` if the dataset has
            no decoder-forcing variables configured.
        """
        df_vars = self.metadata.decoder_forcing_variables
        if not df_vars:
            return None

        values_per_var: dict[str, np.ndarray] = {}
        for source in self.dynamic_forcings_providers:
            needed = [v for v in source.variables if v in df_vars and v not in values_per_var]
            if not needed:
                continue
            # load_forcings_array returns shape (variables, dates, values) for source.variables
            arr = source.load_forcings_array(dates, state)
            source_var_to_idx = {v: i for i, v in enumerate(source.variables)}
            for v in needed:
                values_per_var[v] = arr[source_var_to_idx[v]]  # (dates, values)

        missing = [v for v in df_vars if v not in values_per_var]
        if missing:
            raise RuntimeError(
                f"[{self.dataset_name}] Decoder-forcing variables not found in any "
                f"dynamic-forcings source: {missing}",
            )

        # Sort by ascending data-space position so the tensor matches
        # data_indices.data.input.decoder_forcing, which is also sorted.
        var_to_pos = {v: i for i, v in enumerate(self.metadata.variables)}
        df_vars_sorted = sorted(df_vars, key=lambda v: var_to_pos[v])

        # Stack along feature dim -> (dates, values, n_df) -> (1, dates, 1, values, n_df)
        df_array = np.stack([values_per_var[v] for v in df_vars_sorted], axis=-1)
        df_array = df_array[np.newaxis, :, np.newaxis, ...]
        return torch.from_numpy(df_array).to(self.context.device)

    def add_boundary_forcings_to_input_tensor(
        self,
        input_tensor_torch: "torch.Tensor",
        state: State,
        dates: list[datetime],
        check: BoolArray,
    ) -> "torch.Tensor":
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        sources = self.boundary_forcings_providers
        for source in sources:
            forcings = source.load_forcings_array(dates, state)  # shape: (variables, dates, values)

            forcings = np.swapaxes(forcings, 0, 1)  # shape: (dates, variable, values)

            forcings = np.swapaxes(
                forcings[np.newaxis, :, np.newaxis, ...], -2, -1
            )  # shape: (1, dates, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.context.device)  # Copy to device

            for i in range(min(self.metadata.multi_step_output, self.metadata.multi_step_input)):
                total_mask = np.ix_([0], [-(i + 1)], source.spatial_mask, source.variables_mask)
                input_tensor_torch[total_mask] = forcings[
                    :, -(i + 1), ...
                ]  # Copy forcings to corresponding 'multi_step_input' row

            for n in source.variables_mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(boundary=True, forcing=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "boundary forcings")

        # TO DO: add some consistency checks as above
        return input_tensor_torch

    def _print_input_tensor(self, title: str, input_tensor_torch: dict[str, "torch.Tensor"]) -> None:
        input_tensor_numpy = input_tensor_torch.cpu().numpy()  # (batch, multi_step_input, values, variables)

        assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
        assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

        input_tensor_numpy = np.squeeze(input_tensor_numpy, axis=0)  # Drop the batch dimension
        input_tensor_numpy = np.swapaxes(input_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(
            f"{title} - dataset: `{self.dataset_name}`",
            input_tensor_numpy,
            self._input_tensor_by_name,
            self._input_kinds,
            self._input_units,
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
            f"Output tensor shape={output_tensor_numpy.shape}, NaNs={np.isnan(output_tensor_numpy).sum() / output_tensor_numpy.size: .0%}",
        )

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[-1]):
                variable = self.metadata.output_tensor_index_to_variable[i]
                self._output_tensor_by_name.append(variable)
                if i in self.metadata.prognostic_output_mask:
                    self._output_kinds[variable] = Kind(prognostic=True)
                else:
                    self._output_kinds[variable] = Kind(diagnostic=True)

                self._output_units[variable] = self.metadata.typed_variables[variable].units

        # output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        if len(output_tensor_numpy.shape) == 2:
            output_tensor_numpy = output_tensor_numpy[np.newaxis, ...]  # Add multi_step_input

        output_tensor_numpy = np.swapaxes(output_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(
            title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds, self._output_units
        )

    def _print_tensor(
        self,
        title: str,
        tensor_numpy: FloatArray,
        tensor_by_name: list[str],
        kinds: dict[str, Kind],
        units: dict[str, str],
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
        units : dict
            The units.
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
        table.add_column("Units", justify="left")
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
                str(units.get(v, "N/A")),
                str(kinds.get(v, Kind())),
            )

        console.print()
        console.print(table)
        console.print()

    #########################################################################################################
    def create_constant_computed_forcings(self, variables: list[str], mask: IntArray) -> list["Forcings"]:
        result = ComputedForcings(self, variables, mask)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list[str], mask: IntArray) -> list["Forcings"]:
        result = ComputedForcings(self, variables, mask)
        return [result]

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list["Forcings"]:
        result = ConstantForcings(self, self.constant_forcings_input, variables, mask)
        return [result]

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list["Forcings"]:
        result = CoupledForcings(self, self.dynamic_forcings_input, variables, mask)
        return [result]

    def create_boundary_forcings(self, variables: list[str], mask: IntArray) -> list["Forcings"]:
        result = BoundaryForcings(self, self.boundary_forcings_input, variables, mask)
        return [result]
