# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from ..output import ForwardOutput
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("tee")
class TeeOutput(ForwardOutput):
    """_summary_"""

    def __init__(self, context, *args, outputs=None, output_frequency=None, write_initial_state=None, **kwargs):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        if outputs is None:
            outputs = args

        assert isinstance(outputs, (list, tuple)), outputs
        self.outputs = [create_output(context, output) for output in outputs]

    # We override write_initial_state and write_state
    # so users can configures each levels independently
    def write_initial_state(self, state):
        for output in self.outputs:
            output.write_initial_state(state)

    def write_state(self, state):
        for output in self.outputs:
            output.write_state(state)

    def write_step(self, state):
        raise NotImplementedError("TeeOutput does not support write_step")

    def open(self, state):
        for output in self.outputs:
            output.open(state)

    def close(self):
        for output in self.outputs:
            output.close()

    def __repr__(self):
        return f"TeeOutput({self.outputs})"

    def print_summary(self, depth=0):
        super().print_summary(depth)
        for output in self.outputs:
            output.print_summary(depth + 1)
