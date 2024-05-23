#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from ..checkpoint import Checkpoint
from . import Command


class CheckpointCmd(Command):

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):
        from anemoi.utils.text import dotted_line

        c = Checkpoint(args.path)
        print("num_input_features:", c.num_input_features)
        print("hour_steps:", c.hour_steps)
        result = list(range(0, c.multi_step))
        result = [-s * c.hour_steps for s in result]
        print(sorted(result))
        print("multi_step:", c.multi_step)
        print()
        print("MARS requests:")
        print(dotted_line())
        print("param_sfc:", c.param_sfc)
        print("param_level_pl:", c.param_level_pl)
        print("param_level_ml:", c.param_level_ml)
        print("prognostic_params:", c.prognostic_params)
        print("diagnostic_params:", c.diagnostic_params)
        print("constants_from_input:", c.constants_from_input)
        print("computed_constants:", c.computed_constants)
        print("computed_forcings:", c.computed_forcings)
        print()
        print("imputable variables", c.imputable_variables)


command = CheckpointCmd
