# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

from . import Metadata

LOG = logging.getLogger(__name__)


class DataRequest:
    def __init__(self, metadata):
        self.metadata = metadata

    @property
    def variables(self):
        return self.metadata["variables"]


class ZarrRequest(DataRequest):
    def __init__(self, metadata):
        self.attributes = metadata["attrs"]
        self.request = self.attributes["data_request"]

    @property
    def grid(self):
        return self.request["grid"]

    @property
    def area(self):
        return self.request["area"]

    @property
    def param_sfc(self):
        return self.request["param_level"].get("sfc", [])

    @property
    def param_level_pl(self):
        return self.request["param_level"].get("pl", [])

    @property
    def param_level_ml(self):
        return self.request["param_level"].get("ml", [])

    @property
    def param_step_sfc(self):
        return self.request["param_step"].get("sfc", [])


class Forward(DataRequest):
    @cached_property
    def forward(self):
        return data_request(self.metadata["forward"])

    def __getattr__(self, name):
        return getattr(self.forward, name)


class SubsetRequest(Forward):
    pass


class StatisticsRequest(Forward):
    pass


class RenameRequest(Forward):
    pass


class ConcatRequest(Forward):
    @cached_property
    def forward(self):
        return data_request(self.metadata["datasets"][0])


class JoinRequest(Forward):
    @cached_property
    def forward(self):
        return data_request(self.metadata["datasets"][0])

    @property
    def param_sfc(self):
        result = []
        for dataset in self.metadata["datasets"]:
            for param in data_request(dataset).param_sfc:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_pl(self):
        result = []
        for dataset in self.metadata["datasets"]:
            for param in data_request(dataset).param_level_pl:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_ml(self):
        result = []
        for dataset in self.metadata["datasets"]:
            for param in data_request(dataset).param_level_ml:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_step_sfc(self):
        result = []
        for dataset in self.metadata["datasets"]:
            for param in data_request(dataset).param_step_sfc:
                if param not in result:
                    result.append(param)
        return result


class EnsembleRequest(Forward):
    @cached_property
    def forward(self):
        return data_request(self.metadata["datasets"][0])


class GridRequest(Forward):
    @cached_property
    def forward(self):
        return data_request(self.metadata["datasets"][0])


class SelectRequest(Forward):
    @property
    def param_sfc(self):
        return [x for x in self.forward.param_sfc if x in self.variables]

    @property
    def param_level_pl(self):
        return [x for x in self.forward.param_level_pl if f"{x[0]}_{x[1]}" in self.variables]

    @property
    def param_level_ml(self):
        return [x for x in self.forward.param_level_ml if f"{x[0]}_{x[1]}" in self.variables]

    @property
    def param_step(self):
        return [x for x in self.forward.param_step if x[0] in self.variables]


class DropRequest(SelectRequest):
    pass


def data_request(dataset):
    action = dataset["action"]
    action = action[0].upper() + action[1:].lower() + "Request"
    LOG.debug(f"DataRequest: {action}")
    return globals()[action](dataset)


class Version_0_2_0(Metadata):
    def __init__(self, metadata):
        super().__init__(metadata)
        specific = metadata["dataset"]["specific"]

        self.data_request = data_request(specific)

    @property
    def variables(self):
        return self.data_request.variables

    @cached_property
    def area(self):
        return self.rounded_area(self.data_request.area)

    @property
    def grid(self):
        return self.data_request.grid

    #########################

    @property
    def param_sfc(self):
        param_sfc = self.data_request.param_sfc
        # Remove diagnostic variables
        param_step_sfc = [p[0] for p in self.data_request.param_step_sfc]
        return [p for p in param_sfc if p not in param_step_sfc]

    @property
    def param_level_pl(self):
        param_level_pl = self.data_request.param_level_pl
        params = set([p[0] for p in param_level_pl])
        levels = set([p[1] for p in param_level_pl])
        return sorted(params), sorted(levels)

    @property
    def param_level_ml(self):
        param_level_ml = self.data_request.param_level_ml
        params = set([p[0] for p in param_level_ml])
        levels = set([p[1] for p in param_level_ml])
        return sorted(params), sorted(levels)
