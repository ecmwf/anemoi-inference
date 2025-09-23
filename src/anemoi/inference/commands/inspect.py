# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import sys
import warnings
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from ..checkpoint import Checkpoint
from . import Command

CORE = ("models", "training", "graphs")


SKIP = {"anemoi-training"}

SKIP |= set(sys.stdlib_module_names) | set(sys.builtin_module_names)

PACKAGES = {
    "sklearn": "scikit-learn",
    "attr": "attrs",
    "google-protobuf": "protobuf",
}


class InspectCmd(Command):
    """Inspect the contents of a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint.")
        group = command_parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--variables", action="store_true", help="List the training variables and their categories.")
        group.add_argument(
            "--requirements",
            action="store_true",
            help="Print a Python's requirements.txt based on the versions of the packages used during training.",
        )
        group.add_argument(
            "--datasets", action="store_true", help="Print the arguments passed to anemoi-dataset during training."
        )

        group.add_argument("--dump", action="store_true", help="Dump information from the checkpoint.")

        command_parser.add_argument("--json", action="store_true", help="Output in JSON format (with dump option)")

    def run(self, args: Namespace) -> None:
        """Run the inspect command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """

        c = Checkpoint(args.path)

        if args.validate:
            c.validate_environment()
            return

        if args.variables:
            self.variables(c, args)
            return

        if args.requirements:
            self.requirements(c, args)
            return

        if args.datasets:
            self.datasets(c, args)
            return

        if args.dump:
            self.dump(c, args)
            return

    def dump(self, c: Checkpoint, args: Namespace) -> None:

        if args.json:
            # turn off all other logging so json output can be piped cleanly
            logging.disable(logging.INFO)
            warnings.filterwarnings("ignore")

        def _(f: Callable[[], Any]) -> Any:
            """Wrapper function to handle exceptions.

            Parameters
            ----------
            f : Callable
                The function to be called.

            Returns
            -------
            Any
                The result of the function call or the exception message.
            """
            try:
                return f()
            except Exception as e:
                return str(e)

        data = {}
        for name in sorted(dir(c)):

            if name.startswith("_"):
                continue

            data[name] = _(lambda: getattr(c, name))

        if args.json:
            print(json.dumps(data, indent=None, default=str))
        else:
            for key, value in data.items():
                print(key, ":")
                print("  ", json.dumps(value, indent=4, default=str))
                print()


command = InspectCmd
