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

from ..config import load_config
from ..runners import create_runner
from . import Command

LOG = logging.getLogger(__name__)


def _run(runner, config):
    input = runner.create_input()
    output = runner.create_output()

    # pre_processors = runner.pre_processors
    post_processors = runner.post_processors

    input_state = input.create_input_state(date=config.date)

    output.write_initial_state(input_state)

    for state in runner.run(input_state=input_state, lead_time=config.lead_time):
        for processor in post_processors:
            state = processor.process(state)
        output.write_state(state)

    output.close()


class RunCmd(Command):
    """Run inference from a config yaml file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        config = load_config(args.config, args.overrides, defaults=args.defaults)

        if config.description is not None:
            LOG.info("%s", config.description)

        runner = create_runner(config)

        _run(runner, config)


command = RunCmd
