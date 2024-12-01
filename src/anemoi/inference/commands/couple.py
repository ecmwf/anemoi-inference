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

import yaml

from ..tasks import create_task
from ..transports import create_transport
from . import Command

LOG = logging.getLogger(__name__)


class CoupleCmd(Command):
    """Inspect the contents of a checkpoint file."""

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        config = yaml.safe_load(open(args.config))

        tasks = {name: create_task(name, action) for name, action in config["tasks"].items()}
        for task in tasks.values():
            LOG.info("Task: %s", task)

        transport = create_transport(config["transport"], config["couplings"], tasks)
        LOG.info("Transport: %s", transport)

        transport.start(tasks)
        transport.wait()


command = CoupleCmd
