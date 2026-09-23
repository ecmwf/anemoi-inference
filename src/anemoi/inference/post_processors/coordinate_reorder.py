# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


@post_processor_registry.register("coordinate_reorder")
class CoordinateReorder(Processor):
    """Reverse the operation done to reorder the coordinates of the state arrays, done
    in the `coordinate_reorder` pre-processor.

    Must be used with the `coordinate_reorder` pre-processor.
    """

    def process(self, state: State) -> State:
        """Reorder the coordinates of the state arrays back to their original order.

        Parameters
        ----------
        state : State
            The state dictionary.

        Returns
        -------
        State
            The state dictionary with coordinates reordered to their original order.
        """
        state = state.copy()

        # Retrieve the permutation from the state
        coord_reorder = state.get("_coordinate_reorder", {})
        perm = coord_reorder.get("permutation")

        if perm is None:
            LOG.warning(
                "Missing coordinate reorder permutation in state, did you use `coordinate_reorder` in the pre-processor."
            )
            return state

        for key, field in state["fields"].items():
            data = field[..., perm]
            state["fields"][key] = data

        state["latitudes"] = coord_reorder.get("latitudes", state["latitudes"][perm])
        state["longitudes"] = coord_reorder.get("longitudes", state["longitudes"][perm])
        return state

    def __repr__(self) -> str:
        return "CoordinateReorder()"
