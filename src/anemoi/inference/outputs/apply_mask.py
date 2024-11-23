# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from ..output import Output
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("apply_mask")
class ApplyMaskOutput(Output):
    """_summary_"""

    def __init__(self, context, *, mask, output):
        super().__init__(context)
        self.mask = self.checkpoint.load_supporting_array(mask)
        self.output = create_output(context, output)

    def __repr__(self):
        return f"ApplyMaskOutput({self.mask}, {self.output})"

    def write_initial_state(self, state):
        self.output.write_initial_state(self._apply_mask(state))

    def write_state(self, state):
        self.output.write_state(self._apply_mask(state))

    def _apply_mask(self, state):
        state = state.copy()
        state["fields"] = state["fields"].copy()
        state["latitudes"] = state["latitudes"][self.mask]
        state["longitudes"] = state["longitudes"][self.mask]

        for field in state["fields"]:
            data = state["fields"][field]
            if data.ndim == 1:
                data = data[self.mask]
            else:
                data = data[..., self.mask]
            state["fields"][field] = data

        return state

    def close(self):
        self.output.close()
