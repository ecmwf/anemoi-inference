# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property
from types import MappingProxyType as frozendict

import numpy as np
import torch
from anemoi.utils.checkpoints import load_metadata

from anemoi.inference.forcings import ComputedForcings

from ..checkpoint import Checkpoint
from ..metadata import Metadata
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)

CONSTANT_HIGH_RES_FORCINGS = [
    "cos_latitude",
    "cos_longitude",
    "lsm",
    "sin_latitude",
    "sin_longitude",
    "z",
]

ALL_HIGH_RES_FORCINGS = [
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "lsm",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
    "z",
]

COMPUTED_HIGH_RES_FORCINGS = [forcing for forcing in ALL_HIGH_RES_FORCINGS if forcing not in CONSTANT_HIGH_RES_FORCINGS]


class DsMetadata(Metadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we only need to retrieve from the low res input_0
        self._metadata.data_indices.data.input = self._metadata.data_indices.data.input_0
        self._metadata.data_indices.model.input = self._metadata.data_indices.model.input_0

        # treat all low res inputs as forcings
        self._config.data.forcing = self.low_res_input_variables
        self._metadata.data_indices.data.input.prognostic = []
        self._metadata.data_indices.data.input.diagnostic = []

    @property
    def low_res_input_variables(self):
        return self._metadata.dataset.specific.zip[0]["variables"]

    @property
    def high_res_output_variables(self):
        return self._metadata.dataset.specific.zip[2]["variables"]

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

    def print_indices(self):
        v = {i: v for i, v in enumerate(self.variables)}
        r = {v: k for k, v in self.variable_to_input_tensor_index.items()}
        s = self.output_tensor_index_to_variable

        self._print_indices(
            "Data indices", self._indices.data, dict(input=v, output=v), skip=["output", "input_0", "input_1"]
        )
        self._print_indices(
            "Model indices", self._indices.model, dict(input=r, output=s), skip=["output.full", "input_0", "input_1"]
        )


class DsCheckpoint(Checkpoint):
    @cached_property
    def _metadata(self):
        return DsMetadata(load_metadata(self.path))

    def variables_from_input(self, *, include_forcings):
        # include forcings in initial conditions retrieval
        return super().variables_from_input(include_forcings=True)


@runner_registry.register("downscaling")
class DownscalingRunner(DefaultRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint = DsCheckpoint(self._checkpoint.path)

        self.checkpoint.print_indices()
        self.checkpoint.print_variable_categories()
        self.verbosity = 3

    @cached_property
    def constant_high_res_focings(self):
        file = np.load(self.config.development_hacks.high_res_forcings_npz)
        high_res = np.stack([file[forcing] for forcing in CONSTANT_HIGH_RES_FORCINGS], axis=1)

        return high_res[np.newaxis, np.newaxis, ...]  # shape: (1, 1, values, variables)

    @cached_property
    def high_res_latitudes(self):
        return np.load(self.config.development_hacks.high_res_lat_lon_npz)["latitudes"]

    @cached_property
    def high_res_longitudes(self):
        return np.load(self.config.development_hacks.high_res_lat_lon_npz)["longitudes"]

    def run(self, *, input_state, lead_time):
        if lead_time != 1:
            LOG.info("Forcing lead_time to 1 for downscaling.")
        return super().run(input_state=input_state, lead_time=1)

    def forecast(self, lead_time, input_tensor_numpy, input_state):
        for state in super().forecast(lead_time, input_tensor_numpy, input_state):
            state = state.copy()
            state["latitudes"] = self.high_res_latitudes
            state["longitudes"] = self.high_res_longitudes
            state.pop("_grib_templates_for_output", None)
            yield state

    def predict_step(self, model, input_tensor_torch, input_date, **kwargs):
        low_res_tensor = input_tensor_torch
        high_res_tensor = self._prepare_high_res_input_tensor(input_date)

        LOG.info("Low res tensor shape: %s", low_res_tensor.shape)
        LOG.info("High res tensor shape: %s", high_res_tensor.shape)

        residual_output_tensor = model.predict_step(low_res_tensor, high_res_tensor)
        residual_output_numpy = np.squeeze(residual_output_tensor.cpu().numpy())

        self._print_output_tensor("Residual output tensor", residual_output_numpy)

        if not isinstance(self.config.output, str) and (raw_path := self.config.output.get("raw", {}).get("path")):
            self._save_residual_tensor(residual_output_numpy, f"{raw_path}/output-residuals-o320.npz")

        output_tensor_interp = _prepare_high_res_output_tensor(
            model,
            low_res_tensor[0],  # remove batch dimension
            residual_output_tensor,
            self.checkpoint.variable_to_input_tensor_index,
            self.checkpoint._metadata.high_res_output_variables,
        )

        return output_tensor_interp

    def _prepare_high_res_input_tensor(self, input_date):
        state = {
            "latitudes": self.high_res_latitudes,
            "longitudes": self.high_res_longitudes,
        }

        computed_high_res_forcings = ComputedForcings(self, COMPUTED_HIGH_RES_FORCINGS, []).load_forcings(
            state, input_date
        )
        computed_high_res_forcings = np.squeeze(computed_high_res_forcings, axis=1)  # Drop the dates dimension
        computed_high_res_forcings = np.swapaxes(
            computed_high_res_forcings[np.newaxis, np.newaxis, ...], -2, -1
        )  # shape: (1, 1, values, variables)

        # Merge high res computed and constant forcings so that they are ordered according to ALL_HIGH_RES_FORCINGS
        forcings_dict = {name: None for name in ALL_HIGH_RES_FORCINGS}

        # Fill in the high resolution forcings
        for i, name in enumerate(CONSTANT_HIGH_RES_FORCINGS):
            forcings_dict[name] = self.constant_high_res_focings[..., i]

        # Fill in the computed forcings
        for i, name in enumerate(COMPUTED_HIGH_RES_FORCINGS):
            forcings_dict[name] = computed_high_res_forcings[..., i]

        assert all(forcing is not None for forcing in forcings_dict.values())
        assert set(forcings_dict.keys()) == set(ALL_HIGH_RES_FORCINGS)

        # Stack the forcings in order, shape: (1, 1, values, variables)
        high_res_numpy = np.stack([forcings_dict[name] for name in ALL_HIGH_RES_FORCINGS], axis=-1)

        # print expects shape (step, variables, values)
        self._print_tensor("High res input tensor", np.swapaxes(high_res_numpy[0], -2, -1), forcings_dict.keys(), {})

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
    # interpolate the low res input tensor to high res, and add the residuals to get the final high res output

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
    print("high_res_out is denormalised", high_res_out[..., 0].mean(), high_res_out[..., 0].std())  # 56249 2677

    return high_res_out
