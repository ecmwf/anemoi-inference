#!/usr/bin/env python
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

from ..runner import DefaultRunner
from . import Command

LOGGER = logging.getLogger(__name__)


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("--use-grib-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument("--date", help="Date to use for the request.", default=-1)
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):
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

        runner = DefaultRunner(args.path)

        date = to_datetime(args.date)
        dates = [date + h for h in runner.lagged]

        requests = runner.checkpoint.mars_requests(
            dates=dates,
            expver="0001",
            use_grib_paramid=args.use_grib_paramid,
        )

        input_fields = ekd.from_source("empty")
        for r in requests:
            if r.get("class") in ("rd", "ea"):
                r["class"] = "od"

            if r.get("type") == "fc" and r.get("stream") == "oper" and r["time"] in ("0600", "1800"):
                r["stream"] = "scda"

            r["grid"] = runner.checkpoint.grid
            r["area"] = rounded_area(runner.checkpoint.area)

            print("MARS", _(r))

            input_fields += ekd.from_source("mars", r)

        LOGGER.info("Running the model with the following %s fields, for %s dates", len(input_fields), len(dates))

        runner.run(input_state=input_fields, lead_time=240, device="cuda")


command = RunCmd
