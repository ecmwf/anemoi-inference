# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from argparse import ArgumentParser
from argparse import Namespace

import rich

from anemoi.inference.checkpoint import Checkpoint

from . import Command

LOG = logging.getLogger(__name__)


class VariablesCmd(Command):
    """Used by prepml."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.description = self.__doc__
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument(
            "--indices",
            action="store_true",
        )

    def run(self, args: Namespace) -> None:
        c = Checkpoint(args.path)
        rich.print(f"Checkpoint: {c.path}")
        rich.print("=" * 80)
        c.print_variable_categories(print=rich.print)
        rich.print("=" * 80)
        if args.indices:
            c.print_indices(print=rich.print)


command = VariablesCmd
