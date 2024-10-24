# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from earthkit.data.utils.dates import to_datetime

from .grib import GribInput

LOG = logging.getLogger(__name__)


class MarsInput(GribInput):
    """Get input fields from MARS"""

    def __init__(self, checkpoint, *, use_grib_paramid=False, verbose=True, **kwargs):
        super().__init__(checkpoint, verbose=verbose)
        self.use_grib_paramid = use_grib_paramid
        self.kwargs = kwargs

    def create_input_state(self, *, date):
        if date is None:
            date = to_datetime(-1)
            LOG.warning("MarsInput: `date` parameter not provided, using yesterday's date: %s", date)

        return self._create_input_state(self._retrieve(date))

    def _retrieve(self, date):
        import earthkit.data as ekd

        def rounded_area(area):
            try:
                surface = (area[0] - area[2]) * (area[3] - area[1]) / 180 / 360
                if surface > 0.98:
                    return [90, 0.0, -90, 360]
            except TypeError:
                pass
            return area

        def _(r):
            mars = r.copy()
            for k, v in r.items():
                if isinstance(v, (list, tuple)):
                    mars[k] = "/".join(str(x) for x in v)
                else:
                    mars[k] = str(v)

            return ",".join(f"{k}={v}" for k, v in mars.items())

        dates = [date + h for h in self.checkpoint.lagged]

        requests = self.checkpoint.mars_requests(
            dates=dates,
            expver="0001",
            use_grib_paramid=self.use_grib_paramid,
        )

        if not requests:
            raise ValueError("No MARS requests found in the checkpoint")

        input_fields = ekd.from_source("empty")
        for r in requests:
            if r.get("class") in ("rd", "ea"):
                r["class"] = "od"

            if r.get("type") == "fc" and r.get("stream") == "oper" and r["time"] in ("0600", "1800"):
                r["stream"] = "scda"

            r["grid"] = self.checkpoint.grid
            r["area"] = rounded_area(self.checkpoint.area)

            r.update(self.kwargs)

            print("MARS", _(r))

            input_fields += ekd.from_source("mars", r)

        return input_fields
