# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

from ..checkpoint import Checkpoint
from . import Command

LOG = logging.getLogger(__name__)


class ValidateCmd(Command):
    """Inspect the contents of a checkpoint file."""

    def add_arguments(self, command_parser):

        command_parser.add_argument(
            "--all-packages", action="store_true", help="Check all packages in the environment."
        )

        command_parser.add_argument(
            "--on-difference", choices=["warn", "error", "ignore"], default="warn", help="What to do on difference."
        )

        command_parser.add_argument("--exempt-packages", nargs="*", help="List of packages to exempt from the check.")

        command_parser.add_argument("checkpoint", help="Path to checkpoint file.")

    def run(self, args):
        checkpoint = Checkpoint(args.checkpoint)
        return checkpoint.validate_environment(
            all_packages=args.all_packages,
            on_difference=args.on_difference,
            exempt_packages=args.exempt_packages,
        )


command = ValidateCmd
