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


def _nearest_valid_indices(mask: np.ndarray) -> np.ndarray:
    """Return, for every position, the nearest True index in the mask."""
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("AssignMask requires at least one True value in the mask.")

    indices = np.arange(mask.size)
    pos = np.searchsorted(valid, indices)

    left = valid[np.clip(pos - 1, 0, valid.size - 1)]
    right = valid[np.clip(pos, 0, valid.size - 1)]
    choose_right = np.abs(right - indices) < np.abs(indices - left)
    return np.where(choose_right, right, left)


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
    """

    def __init__(
        self,
        context: Context,
        mask: str,
        fill_value: float = float("NaN"),
        fill_nearest: bool = False,
    ) -> None:
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
        self.npoints = int(np.sum(mask))
        self.fill_value = fill_value
        self.fill_nearest = fill_nearest
        self.nearest_index = _nearest_valid_indices(mask) if fill_nearest else None

    def process(self, state: State) -> State:
        """Assign the state to the mask."""
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

        if self.nearest_index is not None:
            res = res[..., self.nearest_index]

        return res

    def __repr__(self) -> str:
        """Return a string representation of the AssignMask object."""
        extra = ", fill_nearest=True" if self.fill_nearest else ""
        return (
            f"AssignMask(mask={self._maskname}, points={self.npoints}/{self.indexer.size}, "
            f"fill_value={self.fill_value}{extra})"
        )
