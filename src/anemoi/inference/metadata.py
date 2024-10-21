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
from anemoi.transform.variables import Variable
from anemoi.utils.config import find
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import plural
from earthkit.data.indexing.fieldlist import FieldArray

from .patch import PatchMixin

LOG = logging.getLogger(__name__)


class Metadata(PatchMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata):
        self._metadata = metadata

    def dump_masks(self):

        descriptions = {
            # 'data.input.full': "All variables used in training",
            # 'data.input.prognostic': "Variables used in training as prognostic",
        }

        MAX = 20
        variables = self.variables

        def _make_variables_mapping(name):
            data = self._indices["data"]["input"]["full"]
            other = self._indices[name]["input"]["full"]
            assert len(data) == len(other)
            return {j: i for i, j in zip(data, other)}

        variables_mappings = {"data": {i: i for i in range(len(variables))}}

        print()

        for k1, v1 in self._indices.items():

            if k1 not in variables_mappings:
                variables_mappings[k1] = _make_variables_mapping(k1)

            mapping = variables_mappings[k1]

            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    name = ".".join([k1, k2, k3])
                    print(f"✅ {name} ({plural(len(v3), 'value')}) - {descriptions.get(name,'no description')}")
                    size = 0
                    for j, i in enumerate(v3):
                        if j == 0 or size > 100:
                            print()
                            print(".... ", end="")
                            size = 0
                        print(variables[mapping[i]], end=" ")
                        size += len(variables[mapping[i]]) + 1

                    print()
                    print("....", v3[:MAX], "..." if len(v3) > MAX else "")
                    print()

    @property
    def grid(self):
        return self._data_request["grid"]

    @property
    def area(self):
        return self._data_request["area"]

    @property
    def _data_request(self):
        result = find(self._metadata["dataset"], "data_request")
        if len(result) == 0:
            raise ValueError("No data_request found in metadata")

        if len(result) > 1:
            check = ("grid", "area")
            checks = defaultdict(set)
            for r in result:
                for c in check:
                    checks[c].add(str(r.get(c)))

            for c in check:
                if len(checks[c]) > 1:
                    LOG.warning("%s is ambigous: %s", checks[c])

        return result[0]

    ###########################################################################

    @cached_property
    def variable_to_index(self):
        return {v: i for i, v in enumerate(self.variables)}

    @cached_property
    def index_to_variable(self):
        return {i: v for i, v in enumerate(self.variables)}

    @cached_property
    def hour_steps(self):
        return frequency_to_timedelta(self._config_data["frequency"])
        n = self._config_data.get("timestep", self._config_data["frequency"])
        try:
            return int(n)
        except ValueError:
            pass

        return int(n[:-1]) * {"h": 1, "d": 24}[n[-1]]

    @cached_property
    def typed_variables(self):
        """
        Returns a dictionary of strongly typed variables
        """
        return {k: Variable.from_dict(k, v) for k, v in self.variables_metadata.items()}

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

    @property
    def variables(self):
        return self._metadata["dataset"]["variables"]

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

    def filter_and_sort(self, data, dates):
        typed_variables = self.typed_variables

        def _name(field, key, original_metadata):
            param, levelist = original_metadata.get("param"), original_metadata.get("levelist")
            if levelist is None:
                return param
            return f"{param}_{levelist}"

        data = FieldArray([f.copy(name=_name) for f in data])

        variable_from_input = []
        for v in self.variables:
            if typed_variables[v].is_from_input:
                variable_from_input.append(v)

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        data = data.sel(name=variable_from_input, valid_datetime=valid_datetime).order_by("valid_datetime", "name")

        expected = len(variable_from_input) * len(dates)
        if len(data) != expected:
            nvars = plural(len(variable_from_input), "variable")
            ndates = plural(len(dates), "date")
            nfields = plural(expected, "field")
            msg = f"Expected ({nvars}) x ({ndates}) = {nfields}, got {len(data)}"
            LOG.error("%s", msg)
            # TODO: print a report
            raise ValueError(msg)

        assert len(data) == len(variable_from_input) * len(dates)

        return data

    @property
    def number_of_grid_points(self):
        return self._metadata["dataset"]["shape"][-1]

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
        # print("variables_metadata", self.variables_metadata)

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
    def _input_variables_indices(self):
        result = {}
        for i, name in enumerate(self.variables):
            if self.typed_variables[name].is_from_input:
                result[name] = i
        return result

    @cached_property
    def _input_constants_indices(self):
        result = {}
        for i, name in enumerate(self.variables):
            if self.typed_variables[name].is_from_input:
                result[name] = i
        return result

    @cached_property
    def _constants_from_input(self):
        pass

        # return data_mask, model_mask, names

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

    @property
    def variables_metadata(self):
        # print(self._metadata["dataset"])
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
