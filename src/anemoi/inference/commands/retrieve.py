# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import sys

from earthkit.data.utils.dates import to_datetime

from ..config import load_config
from ..inputs.grib import GribInput
from ..inputs.mars import postproc
from ..runners import create_runner
from . import Command


def checkpoint_to_requests(
    checkpoint,
    *,
    date,
    target=None,
    include_forcings=True,
    retrieve_fields_type=None,
    staging_dates=None,
    use_grib_paramid=False,
    extra=None,
    patch_request=None,
    use_scda=False,
):
    # TODO: Move this to the runner

    variables = checkpoint.variables_from_input(include_forcings=include_forcings)
    area = checkpoint.area
    grid = checkpoint.grid

    if retrieve_fields_type is not None:
        selected = set()

        for name, kinds in checkpoint.variable_categories().items():
            if "computed" in kinds:
                continue
            for kind in kinds:
                if retrieve_fields_type.startswith(kind):  # PrepML adds an 's' to the type
                    selected.add(name)

        variables = sorted(selected)

    more = postproc(grid, area)

    for r in extra or []:
        k, v = r.split("=")
        more[k] = v

    if staging_dates:
        dates = set()
        with open(staging_dates) as f:
            for line in f:
                date = to_datetime(line.strip())
                dates.update([date + h for h in checkpoint.lagged])
        dates = sorted(dates)
    else:
        date = to_datetime(date)
        dates = [date + h for h in checkpoint.lagged]

    requests = []
    for r in checkpoint.mars_requests(
        dates=dates,
        variables=variables,
        use_grib_paramid=use_grib_paramid,
        patch_request=patch_request,
        always_split_time=use_scda,
    ):
        r = r.copy()
        if target is not None:
            r["target"] = target
        r.update(more)
        if use_scda:
            _patch_scda(date, r)
        requests.append(r)

    return requests


class RetrieveCmd(Command):
    """Used by prepml."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("config", type=str, help="Path to config file")
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument("--date", type=str, help="Date")
        command_parser.add_argument("--output", type=str, default=None, help="Output file")
        command_parser.add_argument("--staging-dates", type=str, help="Path to a file with staging dates")
        command_parser.add_argument("--extra", action="append", help="Additional request values. Can be repeated")
        command_parser.add_argument("--retrieve-fields-type", type=str, help="Type of fields to retrieve")
        command_parser.add_argument("--use-scda", action="store_true", help="Use scda stream for 6/18 input time")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        config = load_config(args.config, args.overrides, defaults=args.defaults)

        runner = create_runner(config)

        # so that the user does not need to pass --extra target=path when the input file is already in the config
        target = None
        input = runner.create_input()
        if isinstance(input, GribInput) and (path := getattr(input, "path", None)):
            target = path

        requests = checkpoint_to_requests(
            runner.checkpoint,
            date=args.date,
            target=target,
            include_forcings=True,
            extra=args.extra,
            retrieve_fields_type=args.retrieve_fields_type,
            staging_dates=args.staging_dates,
            use_grib_paramid=config.use_grib_paramid,
            patch_request=runner.patch_data_request,
            use_scda=args.use_scda,
        )

        if args.output and args.output != "-":
            f = open(args.output, "w")
        else:
            f = sys.stdout

        json.dump(requests, f, indent=4)


def _patch_scda(base_date, request):
    if base_date.hour not in (6, 18):
        return

    if request.get("class", "od") != "od":
        return
    if request.get("type", "an") not in ("an", "fc"):
        return
    if request.get("stream", "oper") not in ("oper", "scda"):
        return

    request_time = int(request.get("time", 12))
    if request_time > 100:
        request_time = int(request_time / 100)

    if request_time in (6, 18):
        request["stream"] = "scda"


command = RetrieveCmd
