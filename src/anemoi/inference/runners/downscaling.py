# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from datetime import timedelta
from functools import cached_property
from types import MappingProxyType as frozendict
from typing import Optional

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import ComputedForcings
from anemoi.inference.runner import Kind
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State
from anemoi.inference.variables import Variables

from ..checkpoint import Checkpoint
from ..metadata import Metadata
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class DsMetadata(Metadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we only need to retrieve from the low res input_0
        self._indices.data.input = self._indices.data.input_0
        self._indices.model.input = self._indices.model.input_0

        # treat all low res inputs as forcings
        self._config.data.forcing = self.low_res_input_variables
        self._indices.data.input.prognostic = []
        self._indices.data.input.diagnostic = []
        self._indices.model.input.prognostic = []
        self._indices.model.input.diagnostic = []

        # treat all high res outputs as diagnostics
        self._indices.model.output.diagnostic = self._indices.model.output.full
        self._indices.model.output.prognostic = []

    @property
    def low_res_input_variables(self):
        spec = self._metadata.dataset.specific
        if "zip" in spec:
            return spec.zip[0]["variables"]
        elif spec.get("action") == "zip":
            return spec["variables"]
        else:
            raise ValueError(f"Unsupported specific structure: {spec}")

    @property
    def high_res_output_variables(self):
        spec = self._metadata.dataset.specific
        if "zip" in spec:
            # old format
            return spec.zip[2]["variables"]
        elif spec.get("action") == "zip":
            # new format
            return spec["datasets"][2]["variables"]
        else:
            raise ValueError(f"Unsupported specific structure: {spec}")

    @cached_property
    def output_tensor_index_to_variable(self):
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self._indices.model.output.full,
            self._indices.data.output.full,
        )
        return frozendict({k: self.high_res_output_variables[v] for k, v in mapping.items()})

    @cached_property
    def number_of_grid_points(self):
        """Return the number of grid points per fields"""
        if "grid_indices" in self._supporting_arrays:
            return len(self.load_supporting_array("grid_indices"))
        try:
            return self._metadata.dataset.shape[0][-1]
        except AttributeError:
            return self._legacy_number_of_grid_points()

    def print_indices(self, print=LOG.info):
        v = {i: v for i, v in enumerate(self.variables)}
        r = {v: k for k, v in self.variable_to_input_tensor_index.items()}
        s = self.output_tensor_index_to_variable

        self._print_indices(
            "Data indices",
            self._indices.data,
            dict(input=v, output=v),
            skip=["output", "input_0", "input_1"],
        )
        self._print_indices(
            "Model indices",
            self._indices.model,
            dict(input=r, output=s),
            skip=["output.full", "input_0", "input_1"],
        )


class DsCheckpoint(Checkpoint):
    def __init__(self, path: str, *, patch_metadata: dict | None = None):
        # timestep is not read from the metadata, but set by us
        super().__init__(path, patch_metadata=patch_metadata)

    @cached_property
    def _metadata(self):
        return DsMetadata(load_metadata(self.path))


class ZarrDataset:
    def __init__(self, zarr_path: str, forcings: Optional[list[str]] = None):
        if forcings is None:
            forcings = []

        self.constant_forcings: dict[str, FloatArray] = {}

        ds = open_dataset(zarr_path)

        self.lats = ds.latitudes
        self.lons = ds.longitudes
        self.field_shape = ds.field_shape

        var_index_map = {name: ds.name_to_index[name] for name in forcings}
        for var in forcings:
            idx = var_index_map[var]

            if var not in ds.constant_fields:
                continue

            # Only extract the first timestamp since the field is constant
            self.constant_forcings[var] = ds[0, idx, 0, :].flatten()

    def grid_points(self):
        return (self.lats, self.lons)


@runner_registry.register("downscaling")
class DownscalingRunner(DefaultRunner):
    def __init__(
        self,
        config: RunConfiguration,
        time_step: int | str | timedelta,
        ensemble_members: int = 1,
        field_shape: tuple[int, ...] | None = None,
        extra_args: dict | None = None,
        hres_dataset_path: str | None = None,
    ):
        super().__init__(config)

        self.time_step = to_timedelta(time_step)
        self.lead_time = to_timedelta(self.config.lead_time)
        self.write_initial_state = False

        self._checkpoint = DsCheckpoint(
            self._checkpoint.path,
            # some parts of the runner call the checkpoint directly so also overwrite it here
            patch_metadata={"timestep": self.time_step},
        )

        hw = self._checkpoint._metadata._config.hardware
        self.hres_dataset_path = (
            hres_dataset_path if hres_dataset_path is not None else os.path.join(hw.paths.data, hw.files.dataset_y)
        )

        # Need to overwrite this attribute
        self.variables = Variables(self)

        # self.samples = getattr(self.extra_config, "n_samples")
        self.ensemble_members = ensemble_members

        # Overrides for predictions
        self.extra_args = extra_args if extra_args is not None else {}

        # TODO: remove eventually
        self.verbosity = 3

        self._checkpoint.print_indices()
        self._checkpoint.print_variable_categories()

    @property
    def high_res_input(self) -> list[str]:
        return self._checkpoint._metadata._config.dataloader.select_in_hres

    @cached_property
    def constant_high_res_focings_numpy(self) -> FloatArray:
        high_res = np.stack(
            [forcing for forcing in self.hres_dataset.constant_forcings.values()],
            axis=1,
        )

        return high_res[np.newaxis, np.newaxis, ...]  # shape: (1, 1, values, variables)

    @cached_property
    def computed_high_res_forcings(self) -> ComputedForcings:
        # TODO: this breaks if the computed forcings are non constant fields, for example `sr`.
        # But we should not use variables that are not available during production
        computed_forcings = [var for var in self.high_res_input if var not in self.hres_dataset.constant_forcings]
        return ComputedForcings(self, computed_forcings, [])

    @cached_property
    def hres_dataset(self):
        # NOTE: harcoded to read from the checkpoint file
        if "grib" in self.config.output or "netcdf" in self.config.output:
            return ZarrDataset(self.hres_dataset_path, forcings=self.high_res_input)
        else:
            raise Exception("Only grib and netcdf ouputs are available with runner type downscaling.")

    def patch_data_request(self, request):
        # patch initial condition request to include all steps
        request = super().patch_data_request(request)

        lead_hours = int(self.lead_time.total_seconds() // 3600)
        step_hours = int(self.time_step.total_seconds() // 3600)
        request["step"] = f"0/to/{lead_hours}/by/{step_hours}"
        return request

    def copy_prognostic_fields_to_input_tensor(self, input_tensor_torch, y_pred, check):
        # there are no prognostic fields to copy during rollout, so just pass through
        # the full input tensor is retrieved from the input at each step (as dynamic forcings)
        return input_tensor_torch

    def forecast_stepper(self, start_date, lead_time):
        # for downscaling we do a prediction for each step of the input
        steps = (lead_time // self.time_step) + 1  # include step 0

        LOG.info(
            "Lead time: %s, time stepping: %s Forecasting %s steps",
            lead_time,
            self.time_step,
            steps,
        )

        for s in range(steps):
            step = s * self.time_step
            valid_date = start_date + step
            next_date = valid_date + self.time_step
            is_last_step = s == steps - 1
            yield step, valid_date, next_date, is_last_step

    def forecast(self, lead_time: str, input_tensor_numpy: FloatArray, input_state: State):
        for state in super().forecast(lead_time, input_tensor_numpy, input_state):
            state = state.copy()
            state["latitudes"], state["longitudes"] = self.hres_dataset.grid_points()

            if "grib" in self.config.output:
                state["_grib_templates_for_output"] = {name: self.hres_dataset for name in state["fields"].keys()}

            yield state

    def predict_step(self, model, input_tensor_torch, **kwargs):
        date = kwargs["date"]
        step = kwargs["step"]

        input_date = date - step
        low_res_tensor = input_tensor_torch
        high_res_tensor = self._prepare_high_res_input_tensor(input_date)

        LOG.info("Low res tensor shape: %s", low_res_tensor.shape)
        LOG.info("High res tensor shape: %s", high_res_tensor.shape)

        # TODO: is this the correct thing to do to get an ensemble out?
        outputs = []
        for _ in range(self.ensemble_members):
            if self._checkpoint._metadata._config.training.predict_residuals:
                output_tensor = self._predict_from_residuals(model, low_res_tensor, high_res_tensor, **kwargs)
            else:
                output_tensor = self._predict_direct(model, low_res_tensor, high_res_tensor, **kwargs)

            outputs.append(output_tensor)

        # Each output has shape (_, _, _, values, variables)
        # This produces an [_, _, _, n_members, values, variables]
        return torch.stack(outputs, dim=-3)

    def _predict_direct(self, model, low_res_tensor, high_res_tensor, **kwargs):
        output_tensor = model.predict_step(low_res_tensor, high_res_tensor, extra_args=self.extra_args, **kwargs)
        return output_tensor

    def _predict_from_residuals(self, model, low_res_tensor, high_res_tensor, **kwargs):
        residual_output_tensor = model.predict_step(
            low_res_tensor, high_res_tensor, extra_args=self.extra_args, **kwargs
        )
        residual_output_numpy = np.squeeze(residual_output_tensor.cpu().numpy())
        if residual_output_numpy.ndim == 1:
            residual_output_numpy = residual_output_numpy[:, np.newaxis]

        self._print_output_tensor("Residual output tensor", residual_output_numpy)

        if not isinstance(self.config.output, str) and (raw_path := self.config.output.get("raw", {}).get("path")):
            self._save_residual_tensor(residual_output_numpy, f"{raw_path}/output-residuals-o320.npz")

        output_tensor_interp = _prepare_high_res_output_tensor(
            model,
            low_res_tensor[0],  # remove batch dimension
            residual_output_tensor,
            self._checkpoint.variable_to_input_tensor_index,
            self._checkpoint._metadata.high_res_output_variables,
        )

        return output_tensor_interp

    # TODO: make sure this is actually right
    def _prepare_high_res_input_tensor(self, input_date):
        state = {}
        state["latitudes"], state["longitudes"] = self.hres_dataset.grid_points()

        computed_high_res_forcings = self.computed_high_res_forcings.load_forcings_array(input_date, state)

        # Drop the dates dimension
        computed_high_res_forcings = np.squeeze(computed_high_res_forcings, axis=1)

        # Swap last two dimensions so we get shape: (1, 1, values, variables)
        computed_high_res_forcings = np.swapaxes(computed_high_res_forcings[np.newaxis, np.newaxis, ...], -2, -1)

        # Merge high res computed and constant forcings so that
        # they are ordered according to high_res_input
        forcings_dict: dict[str, FloatArray] = {}
        kinds = {}

        # Fill in the constant forcings from cache
        for var, array in self.hres_dataset.constant_forcings.items():
            forcings_dict[var] = array[np.newaxis, np.newaxis, ...]
            kinds[var] = Kind(forcing=True, constant=True)

        # Fill in the computed forcings
        for i, name in enumerate(self.computed_high_res_forcings.variables):
            forcings_dict[name] = computed_high_res_forcings[..., i]
            kinds[name] = Kind(forcing=True, computed=True)

        assert len(forcings_dict) == len(self.high_res_input)
        assert set(forcings_dict.keys()) == set(self.high_res_input)

        # Stack the forcings in order, shape: (1, 1, values, variables)
        high_res_numpy = np.stack([forcings_dict[name] for name in self.high_res_input], axis=-1)

        # print expects shape (step, variables, values)
        self._print_tensor(
            "High res input tensor",
            np.swapaxes(high_res_numpy[0], -2, -1),
            self.high_res_input,
            kinds,
        )

        return torch.from_numpy(high_res_numpy).to(self.device)

    def _save_residual_tensor(self, residual_output_numpy, path):
        # residual_output_numpy.shape: (values, variables)
        np.savez(
            path,
            **{
                f"field_{k}": v
                for k, v in zip(
                    self.checkpoint._metadata.output_tensor_index_to_variable.values(),
                    residual_output_numpy.T,
                )
            },
        )


def _match_tensor_channels(input_name_to_index, output_names):
    """Reorders and selects channels from input tensor to match output tensor structure.
    x_in: Input tensor of shape [batch, n_grid_points, channels]
    """

    common_channels = set(input_name_to_index.keys()) & set(output_names)

    # for each output channel, look for corresponding input channel
    channel_mapping = []
    for channel_name in output_names:
        if channel_name in common_channels:
            input_pos = input_name_to_index[channel_name]
            channel_mapping.append(input_pos)

    # Convert to tensor for indexing
    channel_indices = torch.tensor(channel_mapping)

    return channel_indices


def _prepare_high_res_output_tensor(model, low_res_in, high_res_residuals, input_name_to_index, output_names):
    # interpolate the low res input tensor to high res,
    # and add the residuals to get the final high res output

    matching_channel_indices = _match_tensor_channels(input_name_to_index, output_names)
    print("matching_channel_indices", matching_channel_indices)
    # [64, 25, 36, 46,  3,  0,  1, 16]

    print("low_res_in", low_res_in.shape)  # [1, 40320, 68]

    interp_high_res_in = model.interpolate_down(low_res_in, grad_checkpoint=False)[:, None, None, ...][
        ..., matching_channel_indices
    ]
    print("interp_high_res_in", interp_high_res_in.shape)  # [1, 1, 1, 421120, 8]

    high_res_out = interp_high_res_in + high_res_residuals
    print(
        "interp_high_res_in is denormalised",
        interp_high_res_in[..., 0].mean(),  # 56081.4609
        interp_high_res_in[..., 0].std(),  # 2634.4143
    )

    print(
        "high_res_residuals is denormalised",
        high_res_residuals[..., 0].mean(),  # 168.26504510123863
        high_res_residuals[..., 0].std(),  # 576.3987445776402
    )

    print("high_res_out", high_res_out.shape)  # [1, 1, 1, 421120, 8]
    print(
        "high_res_out is denormalised",
        high_res_out[..., 0].mean(),
        high_res_out[..., 0].std(),
    )  # 56249 2677

    return high_res_out
