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

from icecream import ic

from ..config import load_config
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
        ic(args)
        config = load_config(args.config, args.overrides)

        runner = DefaultRunner(config)

        ic("Create input")
        input = runner.create_input()

        ic("Create input_hres")
        input_hres = runner.create_input_hres()

        ic("Create output")
        output = runner.create_output()

        ic("Create input_0 lres state")
        input_state = input.create_input_state(date=config.date)

        ic("Create input_1 hres state")
        input_hres_state = input_hres.create_input_state(date=config.date)

        ic("Write initial state")
        if config.write_initial_state:
            output.write_initial_state(input_state)
            output.write_initial_state(input_hres_state)

        ic("Run")
        for state in runner.run(
            input_0_state=input_state, input_1_state=input_hres_state
        ):
            output.write_state(state)

        output.close()


command = RunCmd
