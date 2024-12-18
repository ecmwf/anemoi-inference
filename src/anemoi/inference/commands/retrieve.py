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
from ..inputs.mars import postproc
from ..runners.default import DefaultRunner
from . import Command


class RetrieveCmd(Command):
    """Used by prepml."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("config", type=str, help="Path to checkpoint")
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument("--date", type=str, help="Date")
        command_parser.add_argument("--output", type=str, help="Output file")
        command_parser.add_argument("--staging-dates", type=str, help="Path to a file with staging dates")
        command_parser.add_argument("--extra", action="append", help="Additional request values. Can be repeated")
        command_parser.add_argument("--retrieve-fields-type", type=str, help="Type of fields to retrieve")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        config = load_config(args.config, args.overrides, defaults=args.defaults)

        runner = DefaultRunner(config)

        # TODO: Move this to the runner

        variables = runner.checkpoint.variables_from_input(include_forcings=True)
        area = runner.checkpoint.area
        grid = runner.checkpoint.grid

        if args.retrieve_fields_type is not None:
            selected = set()

            for name, kinds in runner.checkpoint.variable_categories().items():
                if "computed" in kinds:
                    continue
                for kind in kinds:
                    if args.retrieve_fields_type.startswith(kind):  # PrepML adds an 's' to the type
                        selected.add(name)

            variables = sorted(selected)

        extra = postproc(grid, area)

        for r in args.extra or []:
            k, v = r.split("=")
            extra[k] = v

        if args.staging_dates:
            dates = []
            with open(args.staging_dates) as f:
                for line in f:
                    dates.append(to_datetime(line.strip()))
        else:
            date = to_datetime(args.date)
            dates = [date + h for h in runner.checkpoint.lagged]

        requests = []
        for r in runner.checkpoint.mars_requests(
            dates=dates,
            variables=variables,
            use_grib_paramid=config.use_grib_paramid,
        ):
            r = r.copy()
            r.update(extra)
            requests.append(r)

        with open(args.output, "w") as f:
            json.dump(requests, f, indent=4)


command = RetrieveCmd
