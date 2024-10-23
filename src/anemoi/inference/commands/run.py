# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging

import numpy as np
from earthkit.data.utils.dates import to_datetime

from ..precisions import PRECISIONS
from ..runners.default import DefaultRunner
from . import Command

LOGGER = logging.getLogger(__name__)


def _dump(state):
    print()
    print("😀", end=" ")
    for key, value in state.items():

        if isinstance(value, datetime.datetime):
            print(f"{key}={value.isoformat()}", end=" ")

        if isinstance(value, (str, float, int, bool, type(None))):
            print(f"{key}={value}", end=" ")

        if isinstance(value, np.ndarray):
            print(f"{key}={value.shape}", end=" ")

    fields = state.get("fields", {})

    print(f"fields={len(fields)}")
    print()

    names = list(fields.keys())
    n = 4

    idx = list(range(0, len(names), len(names) // n))
    idx.append(len(names) - 1)
    idx = sorted(set(idx))

    for i in idx:
        name = names[i]
        field = fields[name]
        min_value = f"min={np.amin(field):g}"
        max_value = f"max={np.amax(field):g}"
        print(f"    {name:8s} shape={field.shape} {min_value:18s} {max_value:18s}")

    print()


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.description = self.__doc__
        command_parser.add_argument("--use-grib-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument("--date", help="Date to use for the request.", default=-1)
        command_parser.add_argument("--device", help="Device to use for the inference.", default="cuda")
        command_parser.add_argument(
            "--precision", help="Precision to use for the inference.", choices=sorted(PRECISIONS.keys())
        )
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):

        args.date = to_datetime(args.date)

        runner = DefaultRunner(args.path, device=args.device, precision=args.precision)
        input_fields = runner.retrieve_input_fields(args.date, args.use_grib_paramid)
        input_state = runner.create_input_state(input_fields)

        _dump(input_state)

        for state in runner.run(input_state=input_state, lead_time=240):
            _dump(state)


command = RunCmd
