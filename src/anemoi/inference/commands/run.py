#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import datetime

from anemoi.utils.dates import as_datetime

from ..runner import DefaultRunner
from . import Command


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("--use-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument("--date", help="Date to use for the request.")
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):
        import earthkit.data as ekd

        runner = DefaultRunner(args.path)

        date = as_datetime(args.date)
        dates = [date + datetime.timedelta(hours=h) for h in runner.lagged]

        requests = runner.checkpoint.mars_requests(
            dates=dates,
            expver="0001",
            use_paramid=args.use_paramid,
        )

        input_fields = ekd.from_source("empty")
        for r in requests:
            if r["class"] == "rd":
                r["class"] = "od"

            r["grid"] = runner.checkpoint.grid
            r["area"] = runner.checkpoint.area

            input_fields += ekd.from_source("mars", r)

        runner.run(input_fields=input_fields, lead_time=240, device="cuda")


command = RunCmd
