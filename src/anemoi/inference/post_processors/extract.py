# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.types import BoolArray
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


class ExtractBase(Processor):
    """Base class for processors that extract data from the state."""

    # this needs to be set in subclasses
    indexer: BoolArray | slice

    def process(self, state: State) -> State:
        """Process the state to extract a subset of points based on the indexer.

        Parameters
        ----------
        state : State
            The state containing fields to be extracted.

        Returns
        -------
        State
            The updated state with extracted fields.
        """
        state = state.copy()
        state["fields"] = state["fields"].copy()

        state["latitudes"] = state["latitudes"][self.indexer]
        state["longitudes"] = state["longitudes"][self.indexer]
        for field in state["fields"]:
            state["fields"][field] = state["fields"][field][self.indexer]

        return state


@post_processor_registry.register("extract_mask")
@main_argument("mask")
class ExtractMask(ExtractBase):
    """Extract a subset of points from the state based on a boolean mask.

    Parameters
    ----------
    context : Any
        The context in which the processor is running.
    mask : str
        Either a path to a `.npy` file containing the boolean mask or
        the name of a supporting array found in the checkpoint.
    """

    def __init__(self, context: Context, mask: str) -> None:
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

    def __repr__(self) -> str:
        """Return a string representation of the ExtractMask object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"ExtractMask({self._maskname}, points={self.npoints}/{self.indexer.size})"


@post_processor_registry.register("extract_slice")
class ExtractSlice(ExtractBase):
    """Extract a subset of points from the state based on a slice.

    Parameters
    ----------
    context : Context
        The context in which the processor is running.
    slice_args : int
        Arguments to create a slice object. This can be a single integer or
        a tuple of integers representing the start, stop, and step of the slice.
    """

    def __init__(self, context: Context, *slice_args: int) -> None:
        super().__init__(context)
        self.indexer = slice(*slice_args)

    def __repr__(self) -> str:
        """Return a string representation of the ExtractSlice object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"ExtractSlice({self.indexer})"
