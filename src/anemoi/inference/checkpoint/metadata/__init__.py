# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import numpy as np
import semantic_version

LOG = logging.getLogger(__name__)


def from_versions(checkpoint_version, dataset_version):
    from .version_0_0_0 import Version_0_0_0
    from .version_0_1_0 import Version_0_1_0
    from .version_0_2_0 import Version_0_2_0

    VERSIONS = {
        ("0.0.0", "0.0.0"): Version_0_0_0,
        ("1.0.0", "0.1.0"): Version_0_1_0,
        ("1.0.0", "0.2.0"): Version_0_2_0,
    }

    version = (
        semantic_version.Version.coerce(checkpoint_version),
        semantic_version.Version.coerce(dataset_version),
    )

    LOG.info("Versions: checkpoint=%s dataset=%s", *version)

    versions = {
        (
            semantic_version.Version.coerce(k[0]),
            semantic_version.Version.coerce(k[1]),
        ): v
        for k, v in VERSIONS.items()
    }

    candidate = None
    for v, klass in sorted(versions.items()):
        if version >= v:
            candidate = klass

    return candidate


class Metadata:
    def __init__(self, metadata):
        self._metadata = metadata

    def to_dict(self):
        return self._metadata

    @classmethod
    def from_metadata(cls, metadata):
        if metadata is None or "dataset" not in metadata:
            metadata = dict(version="0.0.0", dataset=dict(version="0.0.0"))

        if isinstance(metadata["dataset"], list):
            from .patch import list_to_dict

            # Backward compatibility
            metadata["dataset"] = list_to_dict(metadata["dataset"], metadata["config"])

        if "arguments" not in metadata["dataset"]:
            metadata["dataset"]["version"] = "0.1.0"

        # When we changed from ecml_tools to anemoi-datasets, we went back in the
        # versionning
        if metadata["dataset"]["version"] in ("0.1.4", "0.1.7", "0.1.8", "0.1.9"):
            metadata["dataset"]["version"] = "0.2.0"

        klass = from_versions(metadata["version"], metadata["dataset"]["version"])
        return klass(metadata)

    def _find(self, *keys, default=None):
        m = self._metadata
        for key in keys:
            m = m.get(key)
            if m is None:
                return default
        return m

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
            param_level=self.variables,
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
        constants = [
            "cos_latitude",
            "cos_longitude",
            "sin_latitude",
            "sin_longitude",
        ]

        data_mask, model_mask, names = self._forcings(constants)

        LOG.debug("computed_constants data_mask: %s", data_mask)
        LOG.debug("computed_constants model_mask: %s", model_mask)
        LOG.info("Computed constants: %s", names)

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
                "insolation",
            ]
        )

        print("FORCINGS", self._forcing_params())

        constants = set(self._forcing_params()) - set(self.constants_from_input) - set(self.computed_constants)

        if constants - known:
            LOG.warning(f"Unknown computed forcing {constants - known}")

        data_mask, model_mask, names = self._forcings(constants)

        LOG.debug("computed_forcing data_mask: %s", data_mask)
        LOG.debug("computed_forcing model_mask: %s", model_mask)
        LOG.info("Computed forcings: %s", names)

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
        LOG.info("Constants from input: %s", names)

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
        import json

        if "provenance_training" not in self._metadata:
            return

        provenance_training = self._metadata["provenance_training"]

        LOG.error("Training provenance:\n%s", json.dumps(provenance_training, indent=2))

    ###########################################################################

    def graph(self, digraph, nodes, label_maker):
        for kid in self.graph_kids():
            kid.graph(digraph, nodes, label_maker)

    def digraph(self, label_maker=lambda x: dict(label=x.kind)):
        import json

        digraph = ["digraph {"]
        digraph.append("node [shape=box];")
        nodes = {}

        self.graph(digraph, nodes, label_maker)

        for node, label in nodes.items():
            for k, v in label.items():

                if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                    # Keep HTML labels as i
                    label[k] = "<" + json.dumps(v[1:-1])[1:-1] + ">"
                else:
                    # Use json.dumps to escape special characters
                    label[k] = json.dumps(v)

            label = " ".join([f"{k}={v}" for k, v in label.items()])

            digraph.append(f"{node} [{label}];")

        digraph.append("}")
        return "\n".join(digraph)
