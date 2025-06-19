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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


def rounded_area(area: Optional[List[float]]) -> Optional[List[float]]:
    """Round the area to a global extent if the surface is greater than 0.98.

    Parameters
    ----------
    area : Optional[List[float]]
        The area to be rounded.

    Returns
    -------
    Optional[List[float]]
        The rounded area or the original area if no rounding is needed.
    """
    try:
        surface = (area[0] - area[2]) * (area[3] - area[1]) / 180 / 360
        if surface > 0.98:
            return [90, 0.0, -90, 360]
    except TypeError:
        pass
    return area


def grid_is_valid(grid: Optional[Union[str, List[float]]]) -> bool:
    """Check if the grid is valid.

    Parameters
    ----------
    grid : Optional[Union[str, List[float]]]
        The grid to be checked.

    Returns
    -------
    bool
        True if the grid is valid, False otherwise.
    """
    if grid is None:
        return False

    if isinstance(grid, str):
        return True

    try:
        [float(x) for x in grid]
        return True
    except TypeError:
        return False


def area_is_valid(area: Optional[List[float]]) -> bool:
    """Check if the area is valid.

    Parameters
    ----------
    area : Optional[List[float]]
        The area to be checked.

    Returns
    -------
    bool
        True if the area is valid, False otherwise.
    """
    if area is None:
        return False

    if len(area) != 4:
        return False

    try:
        [float(x) for x in area]
        return True
    except TypeError:
        return False


def postproc(
    grid: Optional[Union[str, List[float]]], area: Optional[List[float]]
) -> Dict[str, Union[str, List[float]]]:
    """Post-process the grid and area.

    Parameters
    ----------
    grid : Optional[Union[str, List[float]]]
        The grid to be post-processed.
    area : Optional[List[float]]
        The area to be post-processed.

    Returns
    -------
    Dict[str, Union[str, List[float]]]
        The post-processed grid and area.
    """
    pproc = dict()
    if grid_is_valid(grid):
        pproc["grid"] = grid

    if isinstance(area, str):
        area = [float(x) for x in area.split("/")]

    if area_is_valid(area):
        pproc["area"] = rounded_area(area)

    return pproc


def retrieve(
    requests: List[Dict[str, Any]],
    grid: Optional[Union[str, List[float]]],
    area: Optional[List[float]],
    patch: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Retrieve data from MARS.

    Parameters
    ----------
    requests : List[Dict[str, Any]]
        The list of requests to be retrieved.
    grid : Optional[Union[str, List[float]]]
        The grid for the retrieval.
    area : Optional[List[float]]
        The area for the retrieval.
    patch : Optional[Any], optional
        Optional patch for the request, by default None.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        The retrieved data.
    """
    import earthkit.data as ekd

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
        if r.get("class") in ("rd", "ea"):
            r["class"] = "od"

        # ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC

        if r.get("type") == "fc" and r.get("stream") == "oper" and r["time"] in ("0600", "1800"):
            r["stream"] = "scda"

        r.update(pproc)
        r.update(kwargs)

        if patch:
            r = patch(r)

        LOG.debug("%s", _(r))

        result += ekd.from_source("mars", r)

    return result


@input_registry.register("mars")
class MarsInput(GribInput):
    """Get input fields from MARS."""

    trace_name = "mars"

    def __init__(
        self,
        context: Context,
        *,
        namer: Optional[Any] = None,
        patches: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MarsInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Any]
            Optional namer for the input.
        patches : Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]]
            Optional list of patches for the input.
        **kwargs : Any
            Additional keyword to pass to the request to MARS.
        """
        super().__init__(context, namer=namer)
        self.kwargs = kwargs
        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.kwargs = kwargs
        self.patches = patches or []

    def create_input_state(self, *, date: Optional[Date]) -> State:
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
            LOG.warning("MarsInput: `date` parameter not provided, using yesterday's date: %s", date)

        return self._create_input_state(
            self.retrieve(
                self.variables,
                [date + h for h in self.checkpoint.lagged],
            ),
            variables=self.variables,
            date=date,
        )

    def retrieve(self, variables: List[str], dates: List[Date]) -> Any:
        """Retrieve data for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables to retrieve.
        dates : List[Any]
            The list of dates for which to retrieve the data.

        Returns
        -------
        Any
            The retrieved data.
        """
        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
            patch_request=self.context.patch_data_request,
        )

        if not requests:
            raise ValueError("No requests for %s (%s)" % (variables, dates))

        kwargs = self.kwargs.copy()
        kwargs.setdefault("expver", "0001")
        kwargs.setdefault("grid", self.checkpoint.grid)
        kwargs.setdefault("area", self.checkpoint.area)

        return retrieve(
            requests,
            patch=self.patch,
            **kwargs,
        )

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
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

    def patch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Patch the given request with predefined patches.

        Parameters
        ----------
        request : Dict[str, Any]
            The request to be patched.

        Returns
        -------
        Dict[str, Any]
            The patched request.
        """
        for match, keys in self.patches:
            if all(request.get(k) == v for k, v in match.items()):
                request.update(keys)

        return request
