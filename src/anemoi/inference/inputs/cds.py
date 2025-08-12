# (C) Copyright 2024 Anemoi contributors.
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
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from . import input_registry
from .grib import GribInput
from .mars import postproc

LOG = logging.getLogger(__name__)


def retrieve(
    requests: list[DataRequest],
    grid: str | list[float] | None,
    area: list[float] | None,
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
    area : Optional[List[float]]
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
class CDSInput(GribInput):
    """Get input fields from CDS."""

    trace_name = "cds"

    def __init__(
        self,
        context: Context,
        pre_processors: list[ProcessorConfig] | None = None,
        *,
        dataset: str | dict[str, Any],
        namer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CDSInput.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        pre_processors : Optional[List[ProcessorConfig]], default None
            Pre-processors to apply to the input
        dataset : Union[str, Dict[str, Any]]
            The dataset to use.
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, pre_processors, namer=namer)

        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.dataset = dataset
        self.kwargs = kwargs

    def create_input_state(self, *, date: Date | None) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The created input state.
        """
        if date is None:
            date = to_datetime(-1)
            LOG.warning("CDSInput: `date` parameter not provided, using yesterday's date: %s", date)

        return self._create_input_state(
            self.retrieve(
                self.variables,
                [date + h for h in self.checkpoint.lagged],
            ),
            variables=self.variables,
            date=date,
        )

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve data for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            List of variables to retrieve.
        dates : List[Date]
            List of dates for which to retrieve data.

        Returns
        -------
        Any
            Retrieved data.
        """

        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
            patch_request=self.patch_data_request,
        )

        if not requests:
            raise ValueError(f"No requests for {variables} ({dates})")

        return retrieve(
            requests, self.checkpoint.grid, self.checkpoint.area, dataset=self.dataset, expver="0001", **self.kwargs
        )

    def load_forcings_state(self, *, variables: list[str], dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables for which to load the forcings state.
        dates : List[Date]
            The list of dates for which to load the forcings state.
        current_state : State
            The current state to be updated with the loaded forcings state.

        Returns
        -------
        Any
            The loaded forcings state.
        """
        return self._load_forcings_state(
            self.retrieve(variables, dates), variables=variables, dates=dates, current_state=current_state
        )
