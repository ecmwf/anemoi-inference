# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from earthkit.data.utils.dates import to_datetime

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


def rounded_area(area):
    try:
        surface = (area[0] - area[2]) * (area[3] - area[1]) / 180 / 360
        if surface > 0.98:
            return [90, 0.0, -90, 360]
    except TypeError:
        pass
    return area


def grid_is_valid(grid):
    if grid is None:
        return False

    if isinstance(grid, str):
        return True

    try:
        [float(x) for x in grid]
        return True
    except TypeError:
        return False


def area_is_valid(area):

    if area is None:
        return False

    if len(area) != 4:
        return False

    try:
        [float(x) for x in area]
        return True
    except TypeError:
        return False


def postproc(grid, area):
    pproc = dict()
    if grid_is_valid(grid):
        pproc["grid"] = grid

    if area_is_valid(area):
        pproc["area"] = rounded_area(area)

    return pproc


def retrieve(requests, grid, area, **kwargs):
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

        r.update(pproc)
        r.update(kwargs)

        if (
            r.get("class", "od") == "od"
            and r.get("type") == "fc"
            and r.get("stream") == "oper"
            and r["time"] in ("0600", "1800")
        ):

            r["stream"] = "scda"

        LOG.info("MarsInput: REQUEST  %r", r)

        result += ekd.from_source("mars", r)

    return result


@input_registry.register("mars")
class MarsInput(GribInput):
    """Get input fields from MARS"""

    def __init__(self, context, *, namer=None, **kwargs):
        super().__init__(context, namer=namer)
        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.request = kwargs

        self.request.setdefault("class", "od")
        self.request.setdefault("stream", "oper")
        self.request.setdefault("expver", "0001")
        # self.request.setdefault('type', 'an')

    def create_input_state(self, *, date):
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

    def retrieve(self, variables, dates):

        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
        )

        if not requests:
            raise ValueError("No requests for %s (%s)" % (variables, dates))

        return retrieve(requests, self.checkpoint.grid, self.checkpoint.area, **self.request)

    def template(self, variable, date, **kwargs):
        return self.retrieve([variable], [date])[0]

    def load_forcings_state(self, *, variables, dates, current_state):
        return self._load_forcings_state(
            self.retrieve(variables, dates), variables=variables, dates=dates, current_state=current_state
        )
