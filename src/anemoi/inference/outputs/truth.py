# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from ..context import Context
from ..output import ForwardOutput
from ..output import Output
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("truth")
class TruthOutput(ForwardOutput):
    """Write the truth state for each output state."""

    def __init__(self, context: Context, output: dict, output_frequency=None, write_initial_state=None, **kwargs):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        self.output: Output = create_output(context, output)

    def write_initial_state(self, state):
        self.output.write_initial_state(state)

    def write_state(self, state):
        truth_state = self.context.create_input().create_input_state(date=state["date"])
        reduced_state = self.reduce(truth_state)
        self.output.write_state(reduced_state)

    def write_step(self, state):
        raise NotImplementedError("TruthOutput does not support write_step")

    def open(self, state):
        self.output.open(state)

    def close(self):
        self.output.close()

    def __repr__(self):
        return f"TruthOutput({self.output})"

    def print_summary(self, depth=0):
        super().print_summary(depth)
        self.output.print_summary(depth + 1)
