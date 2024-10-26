# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import json
import logging
import os

import yaml

from ..config import Configuration
from ..inputs.dataset import DatasetInput
from ..inputs.gribfile import GribFileInput
from ..inputs.icon import IconInput
from ..inputs.mars import MarsInput
from ..outputs.gribfile import GribFileOutput
from ..outputs.printer import PrinterOutput
from ..runners.cli import CLIRunner
from . import Command

LOG = logging.getLogger(__name__)


class RunCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        # We use OmegaConf to merge the configuration files and the command line overrides

        with open(args.config) as f:
            config = yaml.safe_load(f)

        for override in args.overrides:
            key, value = override.split("=")
            keys = key.split(".")
            for key in keys[:-1]:
                config = config.setdefault(key, {})
            config[keys[-1]] = value

        config = Configuration(**config)

        LOG.info("Configuration:\n\n%s", json.dumps(config.model_dump(), indent=4, default=str))

        for key, value in config.env.items():
            os.environ[key] = str(value)

        # TODO: Call `Runner.from_config(...)` instead
        runner = CLIRunner(
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
        )

        input, output = self.make_input_output(runner, config)

        input_state = input.create_input_state(date=config.date)

        if config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=240):
            output.write_state(state)

    def make_input_output(self, runner, config):
        # TODO: Use factories

        if config.icon_grid is not None:
            if config.input is None:
                raise ValueError("You must provide an input file to use the ICON plugin")
            input = IconInput(runner, config.input, config.icon_grid, use_grib_paramid=config.use_grib_paramid)
        elif config.input is not None:
            input = GribFileInput(runner, config.input, runner, use_grib_paramid=config.use_grib_paramid)
        elif config.dataset:
            input = DatasetInput(runner)
        else:
            input = MarsInput(runner, use_grib_paramid=config.use_grib_paramid)

        if config.output is not None:
            output = GribFileOutput(config.output, runner, allow_nans=config.allow_nans)
        else:
            output = PrinterOutput(runner)

        return input, output


command = RunCmd
