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
        command_parser.add_argument("--dump", action="store_true", help="Print internal information")

    def run(self, args):

        c = Checkpoint(args.path)

        if args.dump:
            c.dump()
            return

        c.dump()

        print("area:", c.area)
        print("computed_constants_mask:", c.computed_constants_mask)
        print("computed_constants:", c.computed_constants)
        print("computed_forcings_mask:", c.computed_forcings_mask)
        print("computed_forcings:", c.computed_forcings)
        print("constant_data_from_input_mask:", c.constant_data_from_input_mask)
        print("constants_from_input_mask:", c.constants_from_input_mask)
        print("constants_from_input:", c.constants_from_input)
        print("data_to_model:", c.data_to_model)
        print("diagnostic_output_mask:", c.diagnostic_output_mask)
        print("diagnostic_params:", c.diagnostic_params)
        print("grid:", c.grid)
        print("hour_steps:", c.hour_steps)
        print("imputable_variables:", c.imputable_variables)
        print("index_to_variable:", c.index_to_variable)
        print("model_to_data:", c.model_to_data)
        print("multi_step:", c.multi_step)
        print("num_input_features:", c.num_input_features)
        print("operational_config:", c.operational_config)
        print("order_by:", c.order_by)
        print("param_level_ml:", c.param_level_ml)
        print("param_level_pl:", c.param_level_pl)
        print("param_sfc:", c.param_sfc)
        print("precision", c.precision)
        print("prognostic_data_input_mask:", c.prognostic_data_input_mask)
        print("prognostic_input_mask:", c.prognostic_input_mask)
        print("prognostic_output_mask:", c.prognostic_output_mask)
        print("prognostic_params:", c.prognostic_params)
        print("select:", c.select)
        print("variable_to_index:", c.variable_to_index)
        print("variables_with_nans:", c.variables_with_nans)
        print("variables:", c.variables)


command = CheckpointCmd
