# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json

from earthkit.data.utils.dates import to_datetime

from ..config import load_config
from ..runners.default import DefaultRunner
from . import Command


class RetrieveCmd(Command):
    """Used by prepml."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("config", type=str, help="Path to checkpoint")
        command_parser.add_argument("--json", action="store_true", help="Output as JSON")
        command_parser.add_argument("--date", type=str, help="Date")
        command_parser.add_argument("--time", type=str, help="Time")
        command_parser.add_argument("--output", type=str, help="Output file")
        command_parser.add_argument("--staging-dates", type=str, help="Path to a file with staging dates")
        command_parser.add_argument(
            "--requests-extra", action="append", help="Additional request values. Can be repeated"
        )

    def run(self, args):

        config = load_config(args.config, [])

        use_grib_paramid = False  # config.get('use_grib_paramid', False)

        runner = DefaultRunner(config)
        variables = runner.checkpoint.variables_from_input(include_forcings=True)

        requests_extra = dict(area=runner.checkpoint.area, grid=runner.checkpoint.grid)

        for r in args.requests_extra or []:
            k, v = r.split("=")
            requests_extra[k] = v

        if args.staging_dates:
            dates = []
            with open(args.staging_dates) as f:
                for line in f:
                    dates.append(to_datetime(line.strip()))
        else:
            date = to_datetime(args.date)
            assert len(args.time) == 4
            date = date.replace(hour=int(args.time[:2]), minute=int(args.time[2:]))
            dates = [date + h for h in runner.checkpoint.lagged]

        requests = []
        for r in runner.checkpoint.mars_requests(
            dates=dates,
            variables=variables,
            use_grib_paramid=use_grib_paramid,
        ):
            r = r.copy()
            r.update(requests_extra)
            requests.append(r)

        if args.json:
            with open(args.output, "w") as f:
                json.dump(requests, f, indent=4)
            return

        with open(args.output, "w") as f:
            for r in requests:
                req = ["retrieve,target=input.grib"]
                for k, v in r.items():
                    if isinstance(v, (list, tuple)):
                        v = "/".join([str(x) for x in v])
                    req.append(f"{k}={v}")
                r = ",".join(req)
                print(r, file=f)


command = RetrieveCmd
