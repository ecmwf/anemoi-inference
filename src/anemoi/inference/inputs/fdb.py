# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import earthkit.data as ekd

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


def retrieve(
    requests: list[dict[str, Any]],
    configs: dict | None = None,
    **kwargs: Any,
) -> Any:
    """Retrieve data from MARS.

    Parameters
    ----------
    requests : List[dict[str, Any]]
        The list of requests to be retrieved.
    configs : dict, optional
        The FDB configs to use.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        The retrieved data.
    """

    sources: list = []
    for r in requests:
        r.update(kwargs)
        LOG.debug("%s", r)
        sources.append(ekd.from_source("fdb", r, **configs or {}))
    return ekd.from_source("multi", sources)


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
        from_forecast: bool = False,
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
        from_forecast: bool
            Whether to get data from a forecast, i.e. selecting from step, rather than base date.
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
            from_forecast=from_forecast or kwargs.get("type", None) == "fc",
        )
        self.kwargs = kwargs
        self.configs = {"config": fdb_config, "userconfig": fdb_userconfig, "stream": False}
        # NOTE: this is a temporary workaround for #191 thus not documented
        self.param_id_map = kwargs.pop("param_id_map", {})

    def create_input_state(self, *, dates: list[Date], **kwargs) -> State:
        ds = self.retrieve(variables=self.variables, **self._parse_dates(dates))
        return self._create_input_state(ds, variables=None, dates=dates, **kwargs)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        ds = self.retrieve(variables=self.variables, **self._parse_dates(dates))
        return self._load_forcings_state(ds, dates=dates, current_state=current_state)

    def retrieve(self, variables: list[str], dates: list[Date], **kwargs) -> Any:
        requests = self.metadata.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
            patch_request=self.patch_data_request,
        )

        retrieval_kwargs = self.kwargs.copy()
        retrieval_kwargs.update(kwargs)
        # NOTE: this is a temporary workaround for #191
        for request in requests:
            request["param"] = [self.param_id_map.get(p, p) for p in request["param"]]

        LOG.debug("FDB requests: %s", requests)

        return retrieve(
            requests=requests,
            configs=self.configs,
            **retrieval_kwargs,
        )
