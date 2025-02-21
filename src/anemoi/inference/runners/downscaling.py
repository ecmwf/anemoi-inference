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
        self._config.data.forcing = self._metadata["dataset"]["specific"]["zip"][0]["variables"]
        self._metadata.data_indices.data.input.prognostic = []
        self._metadata.data_indices.data.input.diagnostic = []

    @cached_property
    def output_tensor_index_to_variable(self):
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self._indices.model.output.full,
            self._indices.data.output.full,
        )
        variables = self._metadata["dataset"]["specific"]["zip"][2]["variables"]
        return frozendict({k: variables[v] for k, v in mapping.items()})

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
        return super().variables_from_input(include_forcings=True)


@runner_registry.register("downscaling")
class DownscalingRunner(DefaultRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint = DsCheckpoint(self._checkpoint.path)

        self.checkpoint.print_indices()
        self.checkpoint.print_variable_categories()
        self.verbosity = 3

    def forecast(self, lead_time, input_tensor_numpy, input_state):
        for state in super().forecast(lead_time, input_tensor_numpy, input_state):
            state = state.copy()
            state["latitudes"] = self.high_res_latitudes
            state["longitudes"] = self.high_res_longitudes
            state.pop("_grib_templates_for_output", None)
            yield state

    def predict_step(self, model, input_tensor_torch, input_date, **kwargs):
        low_res_tensor = input_tensor_torch
        high_res_tensor = self.prepare_high_res_tensor(input_date)

        LOG.info("Low res tensor shape: %s", low_res_tensor.shape)
        LOG.info("High res tensor shape: %s", high_res_tensor.shape)
        return model.predict_step(low_res_tensor, high_res_tensor)

    @cached_property
    def constant_high_res_focings(self):
        file = np.load(self.config.development_hacks["high_res_forcings_npz"])
        high_res = np.stack([file[forcing] for forcing in CONSTANT_HIGH_RES_FORCINGS], axis=1)

        return high_res[np.newaxis, np.newaxis, ...]  # shape: (1, 1, values, variables)

    @cached_property
    def high_res_latitudes(self):
        return np.load(self.config.development_hacks["high_res_lat_lon_npz"])["latitudes"]

    @cached_property
    def high_res_longitudes(self):
        return np.load(self.config.development_hacks["high_res_lat_lon_npz"])["longitudes"]

    def prepare_high_res_tensor(self, input_date):
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
