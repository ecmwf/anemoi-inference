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


class InspectCmd(Command):

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument("--dump", action="store_true", help="Print internal information")

    def run(self, args):

        c = Checkpoint(args.path)

        if args.dump:
            c.dump()
            return

        def _(f):
            try:
                return f()
            except Exception as e:
                return str(e)

        print("area:", _(lambda: c.area))
        print("computed_constants_mask:", _(lambda: c.computed_constants_mask))
        print("computed_constants:", _(lambda: c.computed_constants))
        print("computed_forcings_mask:", _(lambda: c.computed_forcings_mask))
        print("computed_forcings:", _(lambda: c.computed_forcings))
        print("constant_data_from_input_mask:", _(lambda: c.constant_data_from_input_mask))
        print("constants_from_input_mask:", _(lambda: c.constants_from_input_mask))
        print("constants_from_input:", _(lambda: c.constants_from_input))
        print("data_to_model:", _(lambda: c.data_to_model))
        print("diagnostic_output_mask:", _(lambda: c.diagnostic_output_mask))
        print("diagnostic_params:", _(lambda: c.diagnostic_params))
        print("grid:", _(lambda: c.grid))
        print("hour_steps:", _(lambda: c.hour_steps))
        print("imputable_variables:", _(lambda: c.imputable_variables))
        print("index_to_variable:", _(lambda: c.index_to_variable))
        print("model_to_data:", _(lambda: c.model_to_data))
        print("multi_step:", _(lambda: c.multi_step))
        print("num_input_features:", _(lambda: c.num_input_features))
        print("operational_config:", _(lambda: c.operational_config))
        print("order_by:", _(lambda: c.order_by))
        print("param_level_ml:", _(lambda: c.param_level_ml))
        print("param_level_pl:", _(lambda: c.param_level_pl))
        print("param_sfc:", _(lambda: c.param_sfc))
        print("precision:", _(lambda: c.precision))
        print("prognostic_data_input_mask:", _(lambda: c.prognostic_data_input_mask))
        print("prognostic_input_mask:", _(lambda: c.prognostic_input_mask))
        print("prognostic_output_mask:", _(lambda: c.prognostic_output_mask))
        print("prognostic_params:", _(lambda: c.prognostic_params))
        print("select:", _(lambda: c.select))
        print("variable_to_index:", _(lambda: c.variable_to_index))
        print("variables_with_nans:", _(lambda: c.variables_with_nans))
        print("variables:", _(lambda: c.variables))


command = InspectCmd
