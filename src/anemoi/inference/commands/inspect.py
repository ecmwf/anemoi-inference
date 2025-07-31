# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from ..checkpoint import Checkpoint
from . import Command


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
        command_parser.add_argument(
            "--validate", action="store_true", help="Validate the current virtual environment against the checkpoint"
        )

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

        for name in sorted(dir(c)):

            if name.startswith("_"):
                continue

            print(name, ":")
            print("  ", json.dumps(_(lambda: getattr(c, name)), indent=4, default=str))
            print()


command = InspectCmd
