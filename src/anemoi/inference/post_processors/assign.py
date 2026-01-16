# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry


@post_processor_registry.register("assign_mask")
@main_argument("mask")
class AssignMask(Processor):
    """Assign a mask to the state.

    This processor assigns the state to a larger array using a mask to
    determine the region of assignment. The mask can be provided as a
    boolean numpy array or as a path to a file containing the mask.
    This processor can be seen as the opposite of "extract_mask".
    Instead of extracting a smaller area from a larger one,
    it assigns a state to a larger area using a mask.

    Parameters
    ----------
    context : Context
        The context containing the checkpoint and supporting arrays.
    mask : str
        The name of the mask supporting array or a path to a file containing the mask.
    fill_value : float, optional
        The value to fill the non-assigned area, by default NaN.
    """

    def __init__(self, context: Context, mask: str, fill_value: float = float("NaN")) -> None:
        super().__init__(context)

        self._maskname = mask

        if Path(mask).is_file():
            mask = np.load(mask)
        else:
            mask = context.checkpoint.load_supporting_array(mask)

        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError(
                "Expected the mask to be a boolean numpy array. " f"Got {type(mask)} with dtype {mask.dtype}."
            )

        self.indexer = mask
        self.npoints = np.sum(mask)
        self.fill_value = fill_value

    def process(self, state: State) -> State:
        """Assign the state to the mask.

        Parameters
        ----------
        state : State
            The state dictionary.

        Returns
        -------
        State
            The masked state dictionary.
        """
        state = state.copy()
        state["fields"] = state["fields"].copy()

        state["latitudes"] = self._assign_mask(state["latitudes"])
        state["longitudes"] = self._assign_mask(state["longitudes"])
        for field in state["fields"]:
            state["fields"][field] = self._assign_mask(state["fields"][field])

        return state

    def _assign_mask(self, array: np.ndarray):
        """Logic to assign the array to the mask."""
        shape = array.shape[:-1] + self.indexer.shape
        res = np.full(shape, self.fill_value, dtype=array.dtype)
        if array.ndim == 1:
            res[self.indexer] = array
        else:
            res[..., self.indexer] = array
        return res

    def __repr__(self) -> str:
        """Return a string representation of the AssignMask object.

        Returns
        -------
        str
            A string representation of the AssignMask object.
        """
        return f"AssignMask(mask={self._maskname}, points={self.npoints}/{self.indexer.size}, fill_value={self.fill_value})"
