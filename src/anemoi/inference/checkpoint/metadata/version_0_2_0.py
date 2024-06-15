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
        name = self.__class__.__name__
        return name[:-7] if name.endswith("Request") else name

    @property
    def kind(self):
        kind = self.__class__.__name__
        kind = kind[:-7] if kind.endswith("Request") else kind
        return kind.lower()

    def mars_request(self):

        def _as_list(v):
            if isinstance(v, list):
                return v
            return [v]

        def _as_string(r):
            r = {k: "/".join([str(x) for x in _as_list(v)]) for k, v in r.items() if v}
            return ",".join([f"{k}={v}" for k, v in r.items()])

        r = dict(grid=self.grid, area=self.area)
        yield _as_string(r)

        r = dict(param=_as_list(self.param_sfc))
        yield _as_string(r)

        param, pl = self.param_level_pl
        if param:
            r = dict(param=param, level=pl)
            yield _as_string(r)

        param, ml = self.param_level_ml
        if param:
            r = dict(param=param, level=ml)
            yield _as_string(r)

        nans = self.variables_with_nans
        if nans:
            r = dict(with_nans=nans)
            yield _as_string(r)

    def dump_content(self, indent):

        print()
        print(" " * indent, "-", self)
        for n in self.mars_request():
            print(" " * indent, " ", n)

    @property
    def param_sfc(self):
        param_sfc = self.forward.param_sfc
        param_step_sfc = [p[0] for p in self.forward.param_step_sfc_pairs]
        return [p for p in param_sfc if p not in param_step_sfc]

    @property
    def param_level_pl(self):
        params = set([p[0] for p in self.param_level_pl_pairs])
        levels = set([p[1] for p in self.param_level_pl_pairs])
        return sorted(params), sorted(levels)

    @property
    def param_level_ml(self):
        params = set([p[0] for p in self.param_level_ml_pairs])
        levels = set([p[1] for p in self.param_level_ml_pairs])
        return sorted(params), sorted(levels)

    def graph(self, digraph, nodes, node_maker):
        nodes[f"N{id(self)}"] = node_maker(self)
        for kid in self.graph_kids():
            digraph.append(f"N{id(self)} -> N{id(kid)}")
            kid.graph(digraph, nodes, node_maker)


class ZarrRequest(DataRequest):
    def __init__(self, metadata):
        super().__init__(metadata)
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
    def param_level_pl_pairs(self):
        return self.request["param_level"].get("pl", [])

    @property
    def param_level_ml_pairs(self):
        return self.request["param_level"].get("ml", [])

    @property
    def param_step_sfc_pairs(self):
        return self.request["param_step"].get("sfc", [])

    @property
    def variables_with_nans(self):
        return sorted(self.attributes.get("variables_with_nans", []))

    def dump(self, indent=0):
        self.dump_content(indent)

    def graph_kids(self):
        return []


class Forward(DataRequest):
    @cached_property
    def forward(self):
        return data_request(self.metadata["forward"])

    def __getattr__(self, name):
        return getattr(self.forward, name)

    def dump(self, indent=0):
        self.dump_content(indent)
        self.forward.dump(indent + 2)

    def graph_kids(self):
        return [self.forward]


class SubsetRequest(Forward):
    # Subset in time
    pass


class StatisticsRequest(Forward):
    pass


class RenameRequest(Forward):

    # Drop variables
    # No need to rename anything as self.metadata["variables"] is already up to date

    @property
    def variables_with_nans(self):
        rename = self.metadata["rename"]
        return sorted([rename.get(x, x) for x in self.forward.variables_with_nans])


class MultiRequest(Forward):
    def __init__(self, metadata):
        super().__init__(metadata)
        self.datasets = [data_request(d) for d in metadata["datasets"]]

    @cached_property
    def forward(self):
        return self.datasets[0]

    def dump(self, indent=0):
        self.dump_content(indent)
        for dataset in self.datasets:
            dataset.dump(indent + 2)

    def graph_kids(self):
        return self.datasets


class JoinRequest(MultiRequest):
    """Join variables"""

    @property
    def param_sfc(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_sfc:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_pl_pairs(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_level_pl_pairs:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_level_ml(self):
        result = []
        for dataset in self.datasets_pairs:
            for param in dataset.param_level_ml_pairs:
                if param not in result:
                    result.append(param)
        return result

    @property
    def param_step_sfc_pairs(self):
        result = []
        for dataset in self.datasets:
            for param in dataset.param_step_sfc_pairs:
                if param not in result:
                    result.append(param)
        return result

    @property
    def variables_with_nans(self):
        result = set()
        for dataset in self.datasets:
            result.update(dataset.variables_with_nans)

        return sorted(result)


class ConcatRequest(MultiRequest):
    # Concat in time

    pass


class EnsembleRequest(MultiRequest):
    pass


class MultiGridRequest(MultiRequest):
    @property
    def grid(self):
        grids = [dataset.grid for dataset in self.datasets]
        return grids[0]
        raise NotImplementedError(";".join(str(g) for g in grids))

    @property
    def area(self):
        areas = [dataset.area for dataset in self.datasets]
        return areas[0]
        raise NotImplementedError(";".join(str(g) for g in areas))

    def mars_request(self):
        for d in self.datasets:
            yield from d.mars_request()


class GridsRequest(MultiGridRequest):
    pass


class CutoutRequest(MultiGridRequest):
    pass


class ThinningRequest(Forward):

    @property
    def grid(self):
        return f"thinning({self.forward.grid})"


class SelectRequest(Forward):
    # Select variables

    @property
    def param_sfc(self):
        return [x for x in self.forward.param_sfc if x in self.variables]

    @property
    def param_level_pl_pairs(self):
        return [x for x in self.forward.param_level_pl_pairs if f"{x[0]}_{x[1]}" in self.variables]

    @property
    def param_level_ml_pairs(self):
        return [x for x in self.forward.param_level_ml_pairs if f"{x[0]}_{x[1]}" in self.variables]

    @property
    def param_step_pairs(self):
        return [x for x in self.forward.param_step_pairs if x[0] in self.variables]

    @property
    def param_step_sfc_pairs(self):
        return [x for x in self.forward.param_step_sfc_pairs if x[0] in self.variables]

    @property
    def variables_with_nans(self):
        return [x for x in self.forward.variables_with_nans if x in self.variables]


class DropRequest(SelectRequest):

    # Drop variables
    # No need to drop anything as self.metadata["variables"] is already up to date

    @property
    def variables_with_nans(self):
        return [x for x in self.forward.variables_with_nans if x in self.variables]


def data_request(specific):
    action = specific.pop("action")
    action = action[0].upper() + action[1:].lower() + "Request"
    LOG.debug(f"DataRequest: {action}")
    return globals()[action](specific)


class Version_0_2_0(Metadata, Forward):
    def __init__(self, metadata):
        super().__init__(metadata)
        specific = metadata["dataset"]["specific"]
        self.forward = data_request(specific)

    @cached_property
    def area(self):
        return self.rounded_area(self.forward.area)

    #########################
