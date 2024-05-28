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

    def __repr__(self) -> str:
        return self.__class__.__name__


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

    @property
    def variables_with_nans(self):
        return sorted(self.attributes.get("variables_with_nans", []))

    def dump(self, indent):
        print(" " * indent, self)
        print(" " * indent, self.request)


class Forward(DataRequest):
    @cached_property
    def forward(self):
        return data_request(self.metadata["forward"])

    def __getattr__(self, name):
        return getattr(self.forward, name)

    def dump(self, indent):
        print(" " * indent, self)
        self.forward.dump(indent + 2)


class SubsetRequest(Forward):
    # Subset in time
    pass


class StatisticsRequest(Forward):
    pass


class RenameRequest(Forward):

    @property
    def variables(self):
        raise NotImplementedError()

    @property
    def variables_with_nans(self):
        raise NotImplementedError()


class MultiRequest(Forward):
    def __init__(self, metadata):
        super().__init__(metadata)
        self.datasets = [data_request(d) for d in metadata["datasets"]]

    @cached_property
    def forward(self):
        return self.datasets[0]

    def dump(self, indent):
        print(" " * indent, self)
        for dataset in self.datasets:
            dataset.dump(indent + 2)


class JoinRequest(MultiRequest):
    @property
    def param_sfc(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_sfc:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_pl(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_level_pl:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_ml(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_level_ml:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_step_sfc(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_step_sfc:
                if param not in result:
                    result.append(param)
        return result

    @property
    def variables(self):
        raise NotImplementedError()

    @property
    def variables_with_nans(self):
        result = set()
        for dataset in self.datasets:
            result.update(dataset.variables_with_nans)

        return sorted(result)


class EnsembleRequest(MultiRequest):
    pass


class GridRequest(MultiRequest):
    @property
    def grid(self):
        raise NotImplementedError()

    @property
    def area(self):
        raise NotImplementedError()


class SelectRequest(Forward):
    # Select variables

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

    @property
    def param_step_sfc(self):
        return [x for x in self.forward.param_step_sfc if x[0] in self.variables]

    @property
    def variables_with_nans(self):
        return [x for x in self.forward.variables_with_nans if x in self.variables]


class DropRequest(SelectRequest):

    @property
    def variables(self):
        raise NotImplementedError()

    @property
    def variables_with_nans(self):
        result = set()
        for dataset in self.metadata["datasets"]:
            result.extend(dataset.variables_with_nans)

        return sorted(result)


def data_request(specific):
    action = specific.pop("action")
    action = action[0].upper() + action[1:].lower() + "Request"
    LOG.debug(f"DataRequest: {action}")
    return globals()[action](specific)


class Version_0_2_0(Metadata):
    def __init__(self, metadata):
        super().__init__(metadata)
        specific = metadata["dataset"]["specific"]
        self.data_request = data_request(specific)
        self.data_request.dump(0)

    @property
    def variables(self):
        return self.data_request.variables

    @cached_property
    def area(self):
        return self.rounded_area(self.data_request.area)

    @property
    def grid(self):
        return self.data_request.grid

    @cached_property
    def variables_with_nans(self):
        return self.data_request.variables_with_nans

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
