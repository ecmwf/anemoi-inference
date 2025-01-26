# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

from ..output import ForwardOutput
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("extract_lam")
class ExtractLamOutput(ForwardOutput):
    """_summary_"""

    def __init__(self, context, *, output, lam="lam_0", output_frequency=None, write_initial_state=None):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        if "cutout_mask" in self.checkpoint.supporting_arrays:
            # Backwards compatibility
            mask = self.checkpoint.load_supporting_array("cutout_mask")
            points = slice(None, -np.sum(mask))
        else:
            if lam != "lam_0":
                raise NotImplementedError("Only lam_0 is supported")

            if "lam_1/cutout_mask" in self.checkpoint.supporting_arrays:
                raise NotImplementedError("Only lam_0 is supported")

            mask = self.checkpoint.load_supporting_array(f"{lam}/cutout_mask")
            assert len(mask) == np.sum(mask)
            points = slice(None, np.sum(mask))

        self.points = points
        self.output = create_output(context, output)

    def __repr__(self):
        return f"ExtractLamOutput({self.points}, {self.output})"

    def write_initial_step(self, state):
        # Note: we foreward to 'state', so we write-up options again
        self.output.write_initial_state(self._apply_mask(state))

    def write_step(self, state):
        # Note: we foreward to 'state', so we write-up options again
        self.output.write_state(self._apply_mask(state))

    def _apply_mask(self, state):

        state = state.copy()
        state["fields"] = state["fields"].copy()
        state["latitudes"] = state["latitudes"][self.points]
        state["longitudes"] = state["longitudes"][self.points]

        for field in state["fields"]:
            data = state["fields"][field]
            if data.ndim == 1:
                data = data[self.points]
            else:
                data = data[..., self.points]
            state["fields"][field] = data

        return state

    def close(self):
        self.output.close()

    def print_summary(self, depth=0):
        super().print_summary(depth)
        self.output.print_summary(depth + 1)
