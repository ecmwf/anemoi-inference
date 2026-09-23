# (C) Copyright 2024-2026 Anemoi contributors.
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

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("fdb")
class FDBInput(GribInput):
    """Get input fields from FDB."""

    trace_name = "fdb"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        fdb_config: dict | None = None,
        fdb_userconfig: dict | None = None,
        variables: list[str] | None = None,
        pre_processors: list[ProcessorConfig] | None = None,
        namer: Any | None = None,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the FDB input.

        Parameters
        ----------
        context : Context
            The context for the input.
        metadata : Metadata
            Metadata corresponding to the dataset this input is handling.
        fdb_config : dict, optional
            The FDB config to use.
        fdb_userconfig : dict, optional
            The FDB userconfig to use.
        variables : list[str] | None
            List of variables to be handled by the input, or None for a sensible default variables.
        pre_processors : list[ProcessorConfig], optional
            Pre-processors to apply to the retrieved data.
        namer : Optional[Any]
            Optional namer for the input.
        purpose : str, optional
            The purpose of the input.
        kwargs : dict, optional
            Additional keyword arguments for the request to FDB.
        """
        super().__init__(
            context,
            metadata,
            variables=variables,
            pre_processors=pre_processors,
            purpose=purpose,
            namer=namer,
        )
        self.kwargs = kwargs
        self.configs = {"config": fdb_config, "userconfig": fdb_userconfig}
        # NOTE: this is a temporary workaround for #191 thus not documented
        self.param_id_map = kwargs.pop("param_id_map", {})

    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        date = np.datetime64(date).astype(datetime.datetime)
        dates = [date + h for h in self.metadata.lagged]
        ds = self.retrieve(variables=self.variables, dates=dates)
        res = self._create_input_state(ds, variables=None, date=date, **kwargs)
        return res

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        ds = self.retrieve(variables=self.variables, dates=dates)
        return self._load_forcings_state(ds, dates=dates, current_state=current_state)

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        requests = self.metadata.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
            patch_request=self.patch_data_request,
        )
        requests = [self.kwargs | r for r in requests]
        # NOTE: this is a temporary workaround for #191
        for request in requests:
            request["param"] = [self.param_id_map.get(p, p) for p in request["param"]]

        LOG.debug("FDB requests: %s", requests)
        sources = [ekd.from_source("fdb", request, stream=False, **self.configs) for request in requests]
        ds = ekd.from_source("multi", sources).to_fieldlist()
        return ds
