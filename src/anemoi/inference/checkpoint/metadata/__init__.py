# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from collections import defaultdict
from functools import cached_property

import numpy as np

from .patch import PatchMixin

LOG = logging.getLogger(__name__)


class Metadata(PatchMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata):
        self._metadata = metadata

    # def to_dict(self):
    #     return self._metadata

    # def _find(self, *keys, default=None):
    #     m = self._metadata
    #     for key in keys:
    #         m = m.get(key)
    #         if m is None:
    #             return default
    #     return m

    # Common properties

    ###########################################################################

    @cached_property
    def variable_to_index(self):
        return {v: i for i, v in enumerate(self.variables)}

    @cached_property
    def index_to_variable(self):
        return {i: v for i, v in enumerate(self.variables)}

    @cached_property
    def hour_steps(self):
        n = self._config_data.get("timestep", self._config_data["frequency"])
        try:
            return int(n)
        except ValueError:
            pass

        return int(n[:-1]) * {"h": 1, "d": 24}[n[-1]]

    ###########################################################################
    # Indices

    @cached_property
    def _indices(self):
        return self._metadata["data_indices"]

    @cached_property
    def num_input_features(self):
        return len(self._indices["model"]["input"]["full"])

    @cached_property
    def data_to_model(self):
        data = self._indices["data"]["input"]["full"]
        model = self._indices["model"]["input"]["full"]
        assert len(data) == len(model)
        return {i: j for i, j in zip(data, model)}

    @cached_property
    def model_to_data(self):
        data = self._indices["data"]["input"]["full"]
        model = self._indices["model"]["input"]["full"]
        assert len(data) == len(model)
        return {j: i for i, j in zip(data, model)}

    ###########################################################################
    @property
    def order_by(self):
        # order = self._dataset["order_by"]
        return dict(
            valid_datetime="ascending",
            param_level=self.variables,
            # ensemble=self.checkpoint.ordering('ensemble'),
            remapping={"param_level": "{param}_{levelist}"},
        )

    @property
    def select(self):
        # order = self._dataset["order_by"]
        return dict(
            # valid_datetime="ascending",
            param_level=sorted(
                set(self.variables)
                - set(self.computed_constants)
                - set(self.computed_forcings)
                - set(self.diagnostic_params)
            ),
            # ensemble=self.checkpoint.ordering('ensemble'),
            remapping={"param_level": "{param}_{levelist}"},
        )

    ###########################################################################
    @cached_property
    def _config_data(self):
        """Part of the metadata refers to the model configuration"""
        return self._metadata["config"]["data"]

    @cached_property
    def _config_training(self):
        """Part of the metadata refers to the model configuration"""
        return self._metadata["config"]["training"]

    @cached_property
    def provenance_training(self):
        """Environmental Configuration when trained"""
        return dict(self._metadata.get("provenance_training", {}))

    ###########################################################################
    def _forcings(self, constants):
        forcing = self._indices["data"]["input"]["forcing"]
        data_mask = []
        model_mask = []
        names = []
        for f in forcing:
            if self.index_to_variable[f] in constants:
                data_mask.append(f)
                model_mask.append(self.data_to_model[f])
                names.append(self.index_to_variable[f])

        return data_mask, model_mask, names

    def _forcing_params(self):
        forcing = self._indices["data"]["input"]["forcing"]
        return [self.index_to_variable[f] for f in forcing]

    ###########################################################################

    @cached_property
    def _computed_constants(self):
        print("variables_metadata", self.variables_metadata)

        constants = [
            "cos_latitude",
            "cos_longitude",
            "sin_latitude",
            "sin_longitude",
        ]

        data_mask, model_mask, names = self._forcings(constants)

        LOG.debug("computed_constants data_mask: %s", data_mask)
        LOG.debug("computed_constants model_mask: %s", model_mask)
        LOG.debug("Computed constants: %s", names)

        return data_mask, model_mask, names

    @property
    def computed_constants(self):
        return self._computed_constants[2]

    @property
    def computed_constants_mask(self):
        return self._computed_constants[1]

    ###########################################################################

    @cached_property
    def _computed_forcings(self):
        known = set(
            [
                "cos_julian_day",
                "cos_local_time",
                "sin_julian_day",
                "sin_local_time",
                "insolation",  # Those two are aliases
                "cos_solar_zenith_angle",
            ]
        )

        constants = set(self._forcing_params()) - set(self.constants_from_input) - set(self.computed_constants)

        if constants - known:
            LOG.warning(f"Unknown computed forcing {constants - known}")

        data_mask, model_mask, names = self._forcings(constants)

        LOG.debug("computed_forcing data_mask: %s", data_mask)
        LOG.debug("computed_forcing model_mask: %s", model_mask)
        # LOG.info("Computed forcings: %s", names)

        return data_mask, model_mask, names

    @property
    def computed_forcings(self):
        return self._computed_forcings[2]

    @property
    def computed_forcings_mask(self):
        return self._computed_forcings[1]

    ###########################################################################

    @cached_property
    def _constants_from_input(self):
        """We assume that constants are single level variables"""
        params_sfc = set(self.param_sfc)
        constants = set(self._forcing_params()).intersection(params_sfc)

        data_mask, model_mask, names = self._forcings(constants)

        LOG.debug("constants_from_input: %s", data_mask)
        LOG.debug("constants_from_input: %s", model_mask)
        LOG.debug("Constants from input: %s", names)

        return data_mask, model_mask, names

    @property
    def constants_from_input(self):
        return self._constants_from_input[2]

    @property
    def constants_from_input_mask(self):
        return self._constants_from_input[1]

    @property
    def constant_data_from_input_mask(self):
        return self._constants_from_input[0]

    ###########################################################################

    @cached_property
    def prognostic_input_mask(self):
        return np.array(self._indices["model"]["input"]["prognostic"])

    @cached_property
    def prognostic_data_input_mask(self):
        return np.array(self._indices["data"]["input"]["prognostic"])

    @cached_property
    def prognostic_output_mask(self):
        return np.array(self._indices["model"]["output"]["prognostic"])

    @cached_property
    def diagnostic_output_mask(self):
        return np.array(self._indices["model"]["output"]["diagnostic"])

    @cached_property
    def diagnostic_params(self):
        return [self.index_to_variable[i] for i in self._indices["data"]["input"]["diagnostic"]]

    @cached_property
    def prognostic_params(self):
        return [self.index_to_variable[i] for i in self._indices["data"]["input"]["prognostic"]]

    @cached_property
    def accumulations_params(self):
        # We assume that accumulations are the ones that are forecasts
        return sorted(p[0] for p in self.param_step_sfc_pairs)

    ###########################################################################
    @cached_property
    def precision(self):
        return self._config_training["precision"]

    @cached_property
    def multi_step(self):
        return self._config_training["multistep_input"]

    @cached_property
    def imputable_variables(self):
        return self.variables_with_nans

    def rounded_area(self, area):
        try:
            surface = (area[0] - area[2]) * (area[3] - area[1]) / 180 / 360
            if surface > 0.98:
                return [90, 0.0, -90, 360]
        except TypeError:
            pass
        return area

    def report_loading_error(self):

        if "provenance_training" not in self._metadata:
            return

        provenance_training = self._metadata["provenance_training"]

        LOG.error("Training provenance:\n%s", json.dumps(provenance_training, indent=2))

    ###########################################################################

    @property
    def predict_step_shape(self):
        return (
            1,  # Batch size
            self.multi_step,  # Lagged time steps
            self.number_of_grid_points,  # Grid points
            self.num_input_features,  # Fields
        )

    ###########################################################################
    def summary(self):
        return

        print(f"Prognostics: ({len(self.prognostic_params)})")
        print(sorted(self.prognostic_params))
        print()

        print(f"Diagnostics: ({len(self.diagnostic_params)})")
        print(sorted(self.diagnostic_params))
        print()

        print(f"Retrieved constants: ({len(self.constants_from_input)})")
        print(sorted(self.constants_from_input))
        print()

        print(f"Computed constants: ({len(self.computed_constants)})")
        print(sorted(self.computed_constants))
        print()

        print(f"Computed forcings: ({len(self.computed_forcings)})")
        print(sorted(self.computed_forcings))
        print()

        print(f"Accumulations: ({len(self.accumulations_params)})")
        print(sorted(self.accumulations_params))
        print()

        # print("Select:")
        # print(json.dumps(self.select, indent=2))
        # print()

        # print("Order by:")
        # print(json.dumps(self.order_by, indent=2))
        # print()

    @property
    def variables_metadata(self):
        print(self._metadata["dataset"])
        return self._metadata["dataset"]["variables_metadata"]

    def retrieve_request(self, use_grib_paramid=False):
        from anemoi.utils.grib import shortname_to_paramid
        from earthkit.data.utils.availability import Availability

        keys = ("class", "expver", "type", "stream", "levtype")
        pop = ("date", "time")
        requests = defaultdict(list)
        for variable, metadata in self.variables_metadata.items():
            metadata = metadata.copy()
            if "mars" not in metadata:
                continue

            metadata = metadata["mars"]

            key = tuple(metadata.get(k) for k in keys)
            for k in pop:
                metadata.pop(k, None)

            if use_grib_paramid and "param" in metadata:
                metadata["param"] = shortname_to_paramid(metadata["param"])

            requests[key].append(metadata)

        for reqs in requests.values():

            compressed = Availability(reqs)
            for r in compressed.iterate():
                for k, v in r.items():
                    if isinstance(v, (list, tuple)) and len(v) == 1:
                        r[k] = v[0]
                if r:
                    yield r

    # def patch_metadata(self, metadata, callbacks):
    #     callbacks.patch_zarr(self.attributes, metadata)
    #     return metadata

    # def patch_metadata(self, metadata):
    #     pass
