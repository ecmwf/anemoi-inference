# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any
from typing import List
from typing import Optional

import earthkit.data as ekd
import numpy as np

from ..types import Date
from ..types import State
from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("fdb")
class FDBInput(GribInput):
    """Get input fields from FDB."""

    trace_name = "fdb"

    def __init__(
        self,
        context,
        request_template: dict[str, Any],
        *,
        namer=None,
        fdb_config: dict | None = None,
        fdb_userconfig: dict | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialise the FDB input.

        Parameters
        ----------
        context : dict
            The context runner.
        request_template : dict
            The request template to use.
        namer : optional
            The namer to use for the input.
        fdb_config : dict, optional
            The FDB config to use.
        fdb_userconfig : dict, optional
            The FDB userconfig to use.
        kwargs : dict, optional
            Additional keyword arguments for creating the input state dictionary.
        """
        super().__init__(context, namer=namer)
        self.request_template = dict(request_template)
        self.kwargs = kwargs
        self.configs = {"config": fdb_config, "userconfig": fdb_userconfig}
        self.fdb_config = fdb_config
        self.fdb_userconfig = fdb_userconfig
        self.param_id_map = kwargs.pop("param_id_map", None)  # TODO: this is a temporary workaround for #191
        self.variables = self.checkpoint.variables_from_input(include_forcings=False)

    def create_input_state(self, *, date: Optional[Date]) -> State:
        date = np.datetime64(date).astype(datetime.datetime)
        dates = [date + h for h in self.checkpoint.lagged]
        ds = self.retrieve(variables=self.variables, dates=dates)
        res = self._create_input_state(ds, variables=None, date=date, **self.kwargs)
        return res

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        ds = self.retrieve(variables=variables, dates=dates)
        return self._load_forcings_state(ds, variables=variables, dates=dates, current_state=current_state)

    def retrieve(self, variables: List[str], dates: List[Date]) -> Any:
        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
            patch_request=self.context.patch_data_request,
        )
        requests = [self.request_template | r for r in requests]
        # TODO: this is a temporary workaround for #191
        for request in requests:
            request["param"] = [self.param_id_map.get(p, p) for p in request["param"]]
        sources = [ekd.from_source("fdb", request, stream=False, **self.configs) for request in requests]
        ds = ekd.from_source("multi", sources)
        return ds
