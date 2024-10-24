# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import as_timedelta

from ..inputs.gribfile import GribFileInput
from ..inputs.icon import IconInput
from ..inputs.mars import MarsInput
from ..outputs.gribfile import GribFileOutput
from ..outputs.printer import PrinterOutput
from ..precisions import PRECISIONS
from ..runners.cli import CLIRunner
from . import Command

LOGGER = logging.getLogger(__name__)


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("--use-grib-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument("--date", help="Date to use for the request.")
        command_parser.add_argument("--device", help="Device to use for the inference.", default="cuda")
        command_parser.add_argument("--lead-time", help="Lead time as a timedelta string.", default="10d")
        command_parser.add_argument(
            "--precision", help="Precision to use for the inference.", choices=sorted(PRECISIONS.keys())
        )
        command_parser.add_argument("--input", help="GRIB file to use as input.")
        command_parser.add_argument("--output", help="GRIB file to use as output.")

        command_parser.add_argument(
            "--icon-grid", help="NetCDF containing the ICON grid (e.g. icon_grid_0026_R03B07_G.nc)."
        )

        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):

        if args.date is not None:
            args.date = as_datetime(args.date)

        args.lead_time = as_timedelta(args.lead_time)

        runner = CLIRunner(args.path, device=args.device, precision=args.precision)

        if args.icon_grid is not None:
            if args.input is None:
                raise ValueError("You must provide an input file to use the ICON plugin")
            input = IconInput(args.input, args.icon_grid, runner.checkpoint, use_grib_paramid=args.use_grib_paramid)
        elif args.input is not None:
            input = GribFileInput(args.input, runner.checkpoint, use_grib_paramid=args.use_grib_paramid)
        else:
            input = MarsInput(runner.checkpoint, use_grib_paramid=args.use_grib_paramid)

        if args.output is not None:
            output = GribFileOutput(args.output, runner.checkpoint)
        else:
            output = PrinterOutput(runner.checkpoint)

        input_state = input.create_input_state(date=args.date)

        output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=240):
            output.write_state(state)


command = RunCmd
