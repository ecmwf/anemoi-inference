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

from ..config.couple import CoupleConfiguration
from ..tasks import create_task
from ..transports import create_transport
from . import Command

LOG = logging.getLogger(__name__)

COPY_ATTRIBUTES = (
    "date",
    "lead_time",
    "verbosity",
    "report_error",
)


class CoupleCmd(Command):
    """Couple tasks based on a configuration file."""

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
        """Run the couple command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        if "=" in args.config:
            args.overrides.append(args.config)
            args.config = {}

        config = CoupleConfiguration.load(
            args.config,
            args.overrides,
            defaults=args.defaults,
        )

        if config.description is not None:
            LOG.info("%s", config.description)

        global_config = {}
        for copy in COPY_ATTRIBUTES:
            value = getattr(config, copy, None)
            if value is not None:
                LOG.info("Copy setting to all tasks: %s=%s", copy, value)
                global_config[copy] = value

        tasks = {name: create_task(name, action, global_config=global_config) for name, action in config.tasks.items()}
        for task in tasks.values():
            LOG.info("Task: %s", task)

        transport = create_transport(config.transport, config.couplings, tasks)
        LOG.info("Transport: %s", transport)

        transport.start()
        transport.wait()
        LOG.info("Run complete")


command = CoupleCmd
