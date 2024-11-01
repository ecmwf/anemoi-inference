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
from ..inputs.dataset import DatasetInput
from ..inputs.gribfile import GribFileInput
from ..inputs.icon import IconInput
from ..outputs.gribfile import GribFileOutput
from ..outputs.printer import PrinterOutput
from ..outputs.raw import RawOutput
from ..runners.cutout import CutoutRunner
from ..runners.default import DefaultRunner
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
        RUNNER = CutoutRunner if config.runner == "cutout" else DefaultRunner
        runner = RUNNER(
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
        )

        input, output = self.make_input_output(runner, config)

        input_state = input.create_input_state(date=config.date)

        if config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=config.lead_time):
            output.write_state(state)

    def make_input_output(self, context, config):
        # TODO: Use factories

        if config.icon_grid is not None:
            if config.input is None:
                raise ValueError("You must provide an input file to use the ICON plugin")
            input = IconInput(context, config.input, config.icon_grid, use_grib_paramid=config.use_grib_paramid)
        elif config.input is not None:
            input = GribFileInput(context, config.input, use_grib_paramid=config.use_grib_paramid)
        elif config.dataset:
            input = DatasetInput(context)
        else:
            input = context.mars_input(use_grib_paramid=config.use_grib_paramid)

        if config.output is not None:
            if config.output_type == "grib":
                output = GribFileOutput(context, config.output, allow_nans=config.allow_nans)
            elif config.output_type == "raw":
                output = RawOutput(config.output)
            else:
                raise ValueError("You must set output_typ to either 'grib' or 'raw'. Other formats not yet supported.")
        else:
            output = PrinterOutput(context)

        return input, output


command = RunCmd
