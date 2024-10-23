# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..runner import Runner

LOG = logging.getLogger(__name__)


class CLIRunner(Runner):
    """Running the model from the command line using `anemoi-inference run checkpoint.ckpt`."""

    def retrieve_input_fields(self, date, use_grib_paramid=False):
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
            use_grib_paramid=use_grib_paramid,
        )

        input_fields = ekd.from_source("empty")
        for r in requests:
            if r.get("class") in ("rd", "ea"):
                r["class"] = "od"

            if r.get("type") == "fc" and r.get("stream") == "oper" and r["time"] in ("0600", "1800"):
                r["stream"] = "scda"

            r["grid"] = self.checkpoint.grid
            r["area"] = rounded_area(self.checkpoint.area)

            print("MARS", _(r))

            input_fields += ekd.from_source("mars", r)

        return input_fields
