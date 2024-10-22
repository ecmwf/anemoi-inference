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
from anemoi.utils.config import DotDict
from anemoi.utils.config import find
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import plural
from anemoi.utils.text import blue
from anemoi.utils.text import red
from earthkit.data.indexing.fieldlist import FieldArray

from .legacy import LegacyMixin
from .patch import PatchMixin

LOG = logging.getLogger(__name__)


def compress_list(values):
    if len(values) < 4:
        return values

    ranges = [(values[0], values[1], values[1] - values[0])]
    for i in range(2, len(values)):
        if values[i] - values[i - 1] == ranges[-1][-1]:
            ranges[-1] = (ranges[-1][0], values[i], ranges[-1][-1])
        elif i < len(values) - 1:
            ranges.append((values[i], values[i + 1], values[i + 1] - values[i]))
        else:
            ranges.append((values[i], values[i], 0))

    result = []

    for start, end, diff in ranges:
        if diff == 0:
            result.append(start)
        elif diff == 1:
            if start + 1 == end:
                result.append(start)
                result.append(end)
            else:
                result.append(f"{start}-{end}")
        else:
            for n in range(start, end + 1, diff):
                result.append(n)

    result = ", ".join(str(i) for i in result)
    return f"[{result}]"


class Metadata(PatchMixin, LegacyMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata):
        self._metadata = DotDict(metadata)
        self._masks = {}
        self._variable_mappings = None

    def variable_mappings(self, name):
        # Each `indices` have their own mapping between indices and variables
        # We only know the `data` mapping
        # The others can be computed from the data mapping

        def _make_variables_mapping(name):
            data = self._indices.data.input.full
            other = self._indices[name]["input"]["full"]
            assert len(data) == len(other)
            return {j: i for i, j in zip(data, other)}

        if self._variable_mappings is None:

            self._variable_mappings = {"data": {i: i for i in range(len(self.variables))}}

            for k1 in self._indices.keys():
                if k1 != "data":
                    self._variable_mappings[k1] = _make_variables_mapping(k1)
                    # self._variable_mappings[k1] = {i: i for i in range(len(self.variables))}

        return self._variable_mappings[name]

    def init_masks(self, print=print):

        variables = self.variables

        def _print(name, values, highlight=set()):
            print(f"✅ {name} ({plural(len(values), 'value')})")
            size = 0
            for j, i in enumerate(variables):
                if j == 0 or size > 100:
                    print()
                    print(".... ", end="")
                    size = 0
                if highlight:
                    if i in highlight:
                        print(blue(i), end=" ")
                    else:
                        print(red(i), end=" ")
                else:
                    print(i, end=" ")
                size += len(i) + 1

            print()

        print()
        _print("variables", self.variables)
        print()

        for k1, v1 in self._indices.items():

            mapping = self.variable_mappings(k1)

            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    name = ".".join([k1, k2, k3])
                    try:
                        _print(name, v3, set(variables[mapping[i]] for i in v3))
                    except KeyError:
                        print(f"❌ {name} ({plural(len(v3), 'value')})")
                        for i in v3:
                            print("....", i, mapping.get(i, "???"))
                        print()
                    print(str(compress_list(v3)))
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

    @cached_property
    def typed_variables(self):
        """Returns a strongly typed variables"""
        return {name: Variable.from_dict(name, self.variables_metadata[name]) for name in self.variables}

    ###########################################################################
    # Indices

    @cached_property
    def _indices(self):
        return DotDict(self._metadata["data_indices"])

    @cached_property
    def num_input_features(self):
        return len(self._indices.model.input.full)

    @cached_property
    def data_to_model(self):
        data = self._indices.data.input.full
        model = self._indices.model.input.full
        assert len(data) == len(model)
        return {i: j for i, j in zip(data, model)}

    @cached_property
    def model_to_data(self):
        data = self._indices.data.input.full
        model = self._indices.model.input.full
        assert len(data) == len(model)
        return {j: i for i, j in zip(data, model)}

    @property
    def variables(self):
        return self._metadata.dataset.variables

    @property
    def model_variables(self):
        data = self._indices.data.input.full
        model = self._indices.model.input.full
        print(data, model)

    @property
    def variable_to_input_tensor_index(self):
        data_to_model = self.data_to_model
        return {v: data_to_model.get(i) for i, v in enumerate(self.variables)}

    @property
    def model_input_variables(self):
        data_to_model = self.data_to_model
        return [self.index_to_variable[data_to_model[i]] for i in self._indices.model.input.full]

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

        variable_from_input = [
            v.name for v in typed_variables.values() if v.is_from_input and v.name not in self.diagnostic_params
        ]

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        data = data.sel(name=variable_from_input, valid_datetime=valid_datetime).order_by("name", "valid_datetime")

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

    @cached_property
    def number_of_grid_points(self):
        try:
            return self._metadata.dataset.shape[-1]
        except AttributeError:
            return self._legacy_number_of_grid_points()

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

    @property
    def computed_constants_mask(self):
        """Not time dependent
        Example: cos_latitude, cos_longitude, ...
        """
        forcing = self._indices.data.input.forcing
        typed_variables = self.typed_variables

        return [i for i in forcing if typed_variables[i].is_computed_forcing and typed_variables[i].is_constant_in_time]

    ###########################################################################

    @property
    def computed_forcings_mask(self):
        """Time dependent
        Example: isolation, ...
        """
        forcing = self._indices.data.input.forcing
        typed_variables = self.typed_variables

        return [
            i for i in forcing if typed_variables[i].is_computed_forcing and not typed_variables[i].is_constant_in_time
        ]

    ###########################################################################

    @property
    def constants_from_input_mask(self):
        forcing = self._indices.data.input.full
        typed_variables = self.typed_variables
        return [i for i in forcing if typed_variables[i].is_constant_in_time and typed_variables[i].is_from_input]

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

    @cached_property
    def variables_metadata(self):
        try:
            return self._metadata.dataset.variables_metadata
        except AttributeError:
            return self._legacy_variables_metadata()

    def retrieve_request(self, use_grib_paramid=False):
        from anemoi.utils.grib import shortname_to_paramid
        from earthkit.data.utils.availability import Availability

        keys = ("class", "expver", "type", "stream", "levtype")
        pop = ("date", "time")
        requests = defaultdict(list)
        for variable, metadata in self.variables_metadata.items():

            if "mars" not in metadata:
                continue

            if variable in self.diagnostic_params:
                continue

            metadata = metadata["mars"].copy()

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
