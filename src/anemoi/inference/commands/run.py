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
from argparse import ArgumentParser
from argparse import Namespace

from ..config.run import RunConfiguration
from ..runners import create_runner
from . import Command

LOG = logging.getLogger(__name__)


class RunCmd(Command):
    """Run inference from a config yaml file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument(
            "config",
            help="Path to config file. Can be omitted to pass config with overrides and defaults.",
        )
        command_parser.add_argument("overrides", nargs="*", help="Overrides as key=value")

    def run(self, args: Namespace) -> None:
        """Run the inference command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        if "=" in args.config:
            args.overrides.append(args.config)
            args.config = {}

        config = RunConfiguration.load(
            args.config,
            args.overrides,
            defaults=args.defaults,
        )

        runner = create_runner(config)
        runner.execute()


command = RunCmd
