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

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


def rounded_area(area: Optional[List[float]]) -> Optional[List[float]]:
    try:
        surface = (area[0] - area[2]) * (area[3] - area[1]) / 180 / 360
        if surface > 0.98:
            return [90, 0.0, -90, 360]
    except TypeError:
        pass
    return area


def grid_is_valid(grid: Optional[Union[str, List[float]]]) -> bool:
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
    pproc = dict()
    if grid_is_valid(grid):
        pproc["grid"] = grid

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
    import earthkit.data as ekd

    def _(r):
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
        context: Any,
        *,
        namer: Optional[Any] = None,
        patches: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(context, namer=namer)
        self.kwargs = kwargs
        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.kwargs = kwargs
        self.patches = patches or []

    def create_input_state(self, *, date: Optional[Any]) -> Any:
        if date is None:
            date = to_datetime(-1)
            LOG.warning("MarsInput: `date` parameter not provided, using yesterday's date: %s", date)

        date = to_datetime(date)

        return self._create_state(
            self.retrieve(
                self.variables,
                [date + h for h in self.checkpoint.lagged],
            ),
            variables=self.variables,
            date=date,
        )

    def retrieve(self, variables: List[str], dates: List[Any]) -> Any:

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

        return retrieve(requests, self.checkpoint.grid, self.checkpoint.area, self.patch, **kwargs)

    def load_forcings_state(self, *, variables: List[str], dates: List[Any], current_state: Any) -> Any:
        return self._load_forcings_state(
            self.retrieve(variables, dates), variables=variables, dates=dates, current_state=current_state
        )

    def patch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        for match, keys in self.patches:
            if all(request.get(k) == v for k, v in match.items()):
                request.update(keys)

        return request
