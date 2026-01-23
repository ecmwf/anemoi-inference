# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import json
import logging
import sys
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.metadata import VARIABLE_CATEGORIES
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.variables import Variables

from ..config.run import RunConfiguration
from ..inputs.mars import postproc
from ..runners import create_runner
from . import Command

LOG = logging.getLogger(__name__)


def print_request(verb, request, file=sys.stdout):
    r = [verb]
    for k, v in request.items():
        if not isinstance(v, (list, tuple, set)):
            v = [v]
        v = [str(_) for _ in v]
        v = "/".join(v)
        r.append(f"{k}={v}")

    r = ",\n   ".join(r)
    print(r, file=file)
    print(file=file)


def checkpoint_to_requests(
    checkpoint: Checkpoint,
    *,
    date: Date,
    staging_dates: str | None = None,
    forecast_dates: bool = False,
    use_grib_paramid: bool = False,
    dont_fail_for_missing_paramid: bool = False,
    extra: list[str] | None = None,
    patch_request: Callable[[DataRequest], DataRequest] | None = None,
    use_scda: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    lead_time: Any | None = None,
    time_step: Any | None = None,
) -> list[DataRequest]:
    """Convert a checkpoint to a list of data requests.

    Parameters
    ----------
    checkpoint : Checkpoint
        The checkpoint object containing the necessary data.
    date : Date
        The date for the data request.
    staging_dates : str, optional
        Path to a file with staging dates.
    forecast_dates : bool, optional
        Whether to use forecast dates (for forcings).
    use_grib_paramid : bool, optional
        Whether to use GRIB parameter IDs.
    dont_fail_for_missing_paramid : bool, optional
        Whether to ignore missing parameter IDs.
    extra : list of str, optional
        Additional request values.
    patch_request : Callable[[DataRequest], DataRequest], optional
        Function to patch the data request.
    use_scda : bool, optional
        Whether to use SCDA stream for 6/18 input time.
    include : Optional[List[str]]
        Categories to include.
    exclude : Optional[List[str]]
        Categories to exclude.
    lead_time : Optional[Any]
        The lead time for the data request (used with forecast_dates).
    time_step : Optional[Any]
        The time step for the data request (used with forecast_dates).

    Returns
    -------
    List[DataRequest]
        A list of data requests.
    """
    # TODO: Move this to the runner

    LOG.info("Converting checkpoint to requests")
    LOG.info("Include categories: %s", include)
    LOG.info("Exclude categories: %s", exclude)

    variables = checkpoint.select_variables(include=include, exclude=exclude)

    if not variables:
        return []

    area = checkpoint.area
    grid = checkpoint.grid

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

    if forecast_dates:
        new_dates = set()
        for date in dates:
            d = date
            last = d + lead_time
            while d <= last:
                new_dates.add(d)
                d += time_step

        dates = sorted(new_dates)

    requests = []
    for r in checkpoint.mars_requests(
        dates=dates,
        variables=variables,
        use_grib_paramid=use_grib_paramid,
        patch_request=patch_request,
        always_split_time=use_scda,
        dont_fail_for_missing_paramid=dont_fail_for_missing_paramid,
    ):
        r = r.copy()
        r.update(more)
        if use_scda:
            _patch_scda(r)
        requests.append(r)

    return requests


# Custom type function to parse and validate comma-separated input
def comma_separated_list(value):

    items = value.split(",")
    invalid = set()
    for item in items:
        for bit in item.split("+"):
            if bit not in VARIABLE_CATEGORIES:
                invalid.add(bit)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid value(s): {', '.join(invalid)}. Allowed values are: {', '.join(VARIABLE_CATEGORIES)}"
        )
    return items


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
        command_parser.add_argument("--forecast-dates", action="store_true", help="Use forecast dates (for forcings)")
        command_parser.add_argument("--extra", action="append", help="Additional request values. Can be repeated")
        command_parser.add_argument("--use-scda", action="store_true", help="Use scda stream for 6/18 input time")
        command_parser.add_argument("--use-grib-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument(
            "--dont-fail-for-missing-paramid",
            action="store_true",
            help="Do not fail if a parameter ID is missing.",
        )
        command_parser.add_argument(
            "--include",
            type=comma_separated_list,
            help="Comma-separated list of variable categories to include",
        )
        command_parser.add_argument(
            "--exclude",
            type=comma_separated_list,
            help="Comma-separated list of variable categories to exclude",
        )

        # This is a alias to pairs of include/exclude
        command_parser.add_argument(
            "--input-type",
            default="default-input",
            choices=sorted(Variables.input_types()),
            help="Type of input variables to retrieve.",
        )

        command_parser.add_argument("--mars", action="store_true", help="Write requests for MARS retrieval")
        command_parser.add_argument(
            "--target", default="input.grib", help="Target path for the MARS retrieval requests"
        )
        command_parser.add_argument("--verb", default="retrieve", help="Verb for the MARS retrieval requests")
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
        lead_time = frequency_to_timedelta(config.lead_time)
        time_step = frequency_to_timedelta(runner.checkpoint.timestep)

        if args.staging_dates is None and args.date is None:
            raise ValueError("Either 'date' or 'staging_dates' must be provided.")

        if args.input_type is not None:
            include_exclude = Variables.input_type_to_include_exclude(args.input_type)
            if "include" in include_exclude:
                args.include = sorted(set(args.include or []) | set(include_exclude["include"]))
            if "exclude" in include_exclude:
                args.exclude = sorted(set(args.exclude or []) | set(include_exclude["exclude"]))

        requests = checkpoint_to_requests(
            runner.checkpoint,
            date=args.date,
            extra=args.extra,
            staging_dates=args.staging_dates,
            forecast_dates=args.forecast_dates,
            use_grib_paramid=config.use_grib_paramid or args.use_grib_paramid,
            dont_fail_for_missing_paramid=args.dont_fail_for_missing_paramid,
            patch_request=runner.patch_data_request,
            use_scda=args.use_scda,
            include=args.include if args.include else None,
            exclude=args.exclude if args.exclude else None,
            lead_time=lead_time,
            time_step=time_step,
        )

        if len(requests) > 1:
            requests[0]["target"] = args.target

        if args.output and args.output != "-":
            f = open(args.output, "w")
        else:
            f = sys.stdout

        if args.mars:
            # Write requests in MARS format
            for i, request in enumerate(requests):
                print_request(args.verb, request, file=f)

        else:
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
