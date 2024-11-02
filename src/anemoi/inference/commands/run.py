# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import json
import logging
import os

import yaml

from ..config import Configuration
from ..inputs import create_input
from ..outputs import create_output
from ..runners import runner_registry
from . import Command

LOG = logging.getLogger(__name__)


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        # Load the configuration

        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Apply overrides
        for override in args.overrides:
            key, value = override.split("=")
            keys = key.split(".")
            for key in keys[:-1]:
                config = config.setdefault(key, {})
            config[keys[-1]] = value

        # Load the configuration
        config = Configuration(**config)

        LOG.info("Configuration:\n\n%s", json.dumps(config.model_dump(), indent=4, default=str))

        for key, value in config.env.items():
            os.environ[key] = str(value)

        # TODO: Call `Runner.from_config(...)` instead ???

        runner = runner_registry.create(
            config.runner,
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
        )

        input = create_input(runner, config.input)
        output = create_output(runner, config.output)

        LOG.info("Input: %s", input)
        LOG.info("Output: %s", output)

        input_state = input.create_input_state(date=config.date)

        if config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=config.lead_time):
            output.write_state(state)

        output.close()


command = RunCmd
