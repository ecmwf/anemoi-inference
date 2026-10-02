# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from datetime import datetime
from typing import Any

import earthkit.data as ekd
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import DataRequest
from anemoi.inference.types import ProcessorConfig

from . import input_registry
from .ekd import RequestInput
from .grib import GribInput
from .mars import postproc

LOG = logging.getLogger(__name__)


def retrieve(
    requests: list[DataRequest],
    grid: str | list[float] | None,
    area: list[float] | str | None,
    dataset: str | dict[str, Any],
    **kwargs: Any,
) -> ekd.FieldList:
    """Retrieve data from CDS.

    Parameters
    ----------
    requests : List[Dict[str, Any]]
        List of request dictionaries.
    grid : Optional[Union[str, List[float]]]
        Grid specification.
    area : Optional[Union[List[float], str]]
        Area specification.
    dataset : Union[str, Dict[str, Any]]
        Dataset to use.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        Retrieved data.
    """

    def _(r: DataRequest) -> str:
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    pproc = postproc(grid, area)

    result = ekd.from_source("empty")
    for r in requests:
        if isinstance(dataset, str):
            d = dataset
        elif isinstance(dataset, dict):
            # Get dataset from intersection of keys between request and dataset dict
            search_dataset = dataset.copy()
            while isinstance(search_dataset, dict):
                keys = set(r.keys()).intersection(set(search_dataset.keys()))
                if len(keys) == 0:
                    raise KeyError(
                        f"While searching for dataset, could not find any valid key in dictionary: {r.keys()}, {search_dataset}"
                    )
                key = list(keys)[0]
                if r[key] not in search_dataset[key]:
                    if "*" in search_dataset[key]:
                        search_dataset = search_dataset[key]["*"]
                        continue

                    raise KeyError(
                        f"Dataset dictionary does not contain key {r[key]!r} in {key!r}: {dict(search_dataset[key])}."
                    )
                search_dataset = search_dataset[key][r[key]]

            d = search_dataset

        r.update(pproc)
        r.update(kwargs)

        LOG.debug("%s", _(r))

        result += ekd.from_source("cds", d, r)

    return result


@input_registry.register("cds")
class CDSInput(GribInput, RequestInput):
    """Get input fields from CDS."""

    trace_name = "cds"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        variables: list[str] | None = None,
        pre_processors: list[ProcessorConfig] | None = None,
        dataset: str | dict[str, Any],
        namer: Any | None = None,
        purpose: str | None = None,
        from_forecast: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the CDSInput.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        metadata : Metadata
            Metadata corresponding to the dataset this input is handling.
        variables : list[str] | None
            List of variables to be handled by the input, or None for a sensible default variables.
        pre_processors : Optional[List[ProcessorConfig]], default None
            Pre-processors to apply to the input
        dataset : Union[str, Dict[str, Any]]
            The dataset to use.
        namer : Optional[Any]
            Optional namer for the input.
        purpose : Optional[str]
            The purpose of the input (e.g., 'forcings', 'constants'). Used for debugging and logging.
        from_forecast: bool
            Whether to get data from a forecast, i.e. selecting from step, rather than base date.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(
            context,
            metadata,
            variables=variables,
            pre_processors=pre_processors,
            namer=namer,
            purpose=purpose,
            from_forecast=from_forecast,
        )

        self.dataset = dataset
        self.kwargs = kwargs

    def _retrieve(self, requests: list[DataRequest], **kwargs: Any) -> Any:
        """Retrieve data from CDS for the given requests.

        Parameters
        ----------
        requests : list[DataRequest]
            The list of requests to retrieve.
        **kwargs : Any
            Additional keyword arguments to pass to the retrieval function.

        Returns
        -------
        Any
            Retrieved data.
        """
        retrieval_kwargs = self.kwargs.copy()
        retrieval_kwargs.update(kwargs)

        return retrieve(
            requests,
            self.metadata.grid,
            self.metadata.area,
            dataset=self.dataset,
            expver="0001",
            **retrieval_kwargs,
        )

    def default_initial_date(self) -> datetime:
        # yesterday midnight
        return to_datetime(-1)
