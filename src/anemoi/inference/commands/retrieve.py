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
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from earthkit.data.utils.dates import to_datetime

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date

from ..config.run import RunConfiguration
from ..inputs.grib import GribInput
from ..inputs.mars import postproc
from ..runners import create_runner
from . import Command


def checkpoint_to_requests(
    checkpoint: Checkpoint,
    *,
    date: Date,
    target: str | None = None,
    include_forcings: bool = True,
    retrieve_fields_type: str | None = None,
    staging_dates: str | None = None,
    use_grib_paramid: bool = False,
    extra: list[str] | None = None,
    patch_request: Callable[[DataRequest], DataRequest] | None = None,
    use_scda: bool = False,
) -> list[DataRequest]:
    """Convert a checkpoint to a list of data requests.

    Parameters
    ----------
    checkpoint : Checkpoint
        The checkpoint object containing the necessary data.
    date : Date
        The date for the data request.
    target : str, optional
        The target path for the data request.
    include_forcings : bool, optional
        Whether to include forcings in the data request.
    retrieve_fields_type : str, optional
        The type of fields to retrieve.
    staging_dates : str, optional
        Path to a file with staging dates.
    use_grib_paramid : bool, optional
        Whether to use GRIB parameter IDs.
    extra : list of str, optional
        Additional request values.
    patch_request : Callable[[DataRequest], DataRequest], optional
        Function to patch the data request.
    use_scda : bool, optional
        Whether to use SCDA stream for 6/18 input time.

    Returns
    -------
    List[DataRequest]
        A list of data requests.
    """
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
            _patch_scda(r)
        requests.append(r)

    return requests


class RetrieveCmd(Command):
    """Used by prepml."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.description = self.__doc__
        command_parser.add_argument(
            "config",
            type=str,
            help="Path to config file. Can be omitted to pass config with overrides and defaults.",
        )
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument("--date", type=str, help="Date")
        command_parser.add_argument("--output", type=str, default=None, help="Output file")
        command_parser.add_argument("--staging-dates", type=str, help="Path to a file with staging dates")
        command_parser.add_argument("--extra", action="append", help="Additional request values. Can be repeated")
        command_parser.add_argument("--retrieve-fields-type", type=str, help="Type of fields to retrieve")
        command_parser.add_argument("--use-scda", action="store_true", help="Use scda stream for 6/18 input time")
        command_parser.add_argument("overrides", nargs="*", help="Overrides as key=value")

    def run(self, args: Namespace) -> None:
        """Run the retrieve command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        if "=" in args.config:
            args.overrides.append(args.config)
            args.config = {}

        config: RunConfiguration = RunConfiguration.load(args.config, args.overrides, defaults=args.defaults)

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


def _patch_scda(request: dict[str, Any]) -> None:
    """Patch the SCDA stream in the request if necessary.
    ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC.

    Parameters
    ----------
    request : dict
        The request dictionary to be patched.
    """

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
