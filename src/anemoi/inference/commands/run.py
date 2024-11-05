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
import os

import yaml

from ..config import Configuration
from ..runners.default import DefaultRunner
from . import Command

LOG = logging.getLogger(__name__)


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def get_config(self, args):

        # Load the configuration

        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Apply overrides
        for override in args.overrides:
            path = config
            key, value = override.split("=")
            keys = key.split(".")
            for key in keys[:-1]:
                path = path.setdefault(key, {})
            path[keys[-1]] = value

        # Load the configuration
        config = Configuration(**config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config

    def run(self, args):

        config = self.get_config(args)

        runner = DefaultRunner(config)

        input = runner.create_input()
        output = runner.create_output()

        input_state = input.create_input_state(date=config.date)

        if config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=config.lead_time):
            output.write_state(state)

        output.close()


command = RunCmd
