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


class Version_0_1_0(Metadata):
    def __init__(self, metadata):
        super().__init__(metadata)
        self.patch_metadata()

    @cached_property
    def _dataset(self):
        """
        Part of the metadata that refers to the zarr dataset
        """
        return self._metadata["dataset"]

    @cached_property
    def param_sfc(self):
        param_level_sfc = self._dataset["data_request"]["param_level"]["sfc"]
        param_step_sfc = self._dataset["data_request"]["param_step"]["sfc"]
        fc_param = [param for param, step in param_step_sfc if step != 0]
        return sorted([p for p in param_level_sfc if p not in fc_param])

    @cached_property
    def param_level_pl(self):
        if "pl" not in self._dataset["data_request"]["param_level"]:
            return [], []
        result = self._dataset["data_request"]["param_level"]["pl"]
        return sorted(set([r[0] for r in result])), sorted(set([r[1] for r in result]))

    @cached_property
    def param_level_ml(self):
        if "ml" not in self._dataset["data_request"]["param_level"]:
            return [], []
        result = self._dataset["data_request"]["param_level"]["ml"]
        return sorted(set([r[0] for r in result])), sorted(set([r[1] for r in result]))

    @cached_property
    def param_step(self):
        return []

    @cached_property
    def grid(self):
        return self._dataset["data_request"]["grid"]

    @cached_property
    def area(self):
        return self.rounded_area(self._dataset["data_request"]["area"])

    @cached_property
    def variables(self):
        return self._dataset["variables"]

    ###########################################################################

    def patch_metadata(self):
        drop = self._find("config", "dataloader", "training", "drop")
        if drop is not None:
            variables = self._find("dataset", "variables", default=[])
            pl = self._find("dataset", "data_request", "param_level", "pl", default=[])
            ml = self._find("dataset", "data_request", "param_level", "ml", default=[])
            sfc = self._find("dataset", "data_request", "param_level", "sfc", default=[])
            for var in drop:
                if var in variables:
                    variables.remove(var)
                if var in sfc:
                    sfc.remove(var)
                if "_" in var:
                    param, level = var.split("_")
                    level = int(level)
                    if [param, level] in pl:
                        pl.remove([param, level])
                    if [param, level] in ml:
                        ml.remove([param, level])

    @property
    def variables_with_nans(self):
        return []

    def dump(self, indent=0):
        print("Version_0_1_0: Not implemented")

    def graph_kids(self):
        from .version_0_2_0 import ZarrRequest

        dataset = self._dataset.copy()
        if "attrs" not in dataset:
            dataset["attrs"] = dataset.copy()

        return [ZarrRequest(dataset)]
