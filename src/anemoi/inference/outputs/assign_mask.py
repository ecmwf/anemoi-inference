# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

import numpy as np

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..output import ForwardOutput
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("assign_mask")
class AssignMask(ForwardOutput):
    """Assign the current output to larger area using a mask.

    This operation can be seen as the opposite of "extract_mask".
    Instead of extracting a smaller area from a larger one,
    it assigns the current output to a larger area using a mask.
    This is useful when you want to restore the original state of the model
    after applying a mask to it. The portion of the state that is not
    covered by the mask will be set to a fill value (NaN by default).

    Parameters
    ----------
    context : dict
        The context dictionary.
    output : dict
        The output configuration dictionary.
    mask : str
        The mask supporting array name.
    fill_value : float, optional
        The fill value to use for the masked area, by default np.nan.
    output_frequency : int, optional
        The frequency of output, by default None.
    write_initial_state : bool, optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self,
        context: Context,
        *,
        output: Configuration,
        mask: str,
        fill_value: float = np.nan,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        super().__init__(context, output, output_frequency=output_frequency, write_initial_state=write_initial_state)

        if mask not in self.checkpoint.supporting_arrays:
            raise ValueError(f"Assignment mask '{mask}' not found in supporting arrays.")

        self.mask = self.checkpoint.load_supporting_array(mask)
        self.fill_value = fill_value

    def __repr__(self) -> str:
        """Return a string representation of the ExtractLamOutput object."""
        return f"AssignMask({self.output})"

    def modify_state(self, state: State) -> State:
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

        state["latitudes"] = self._assign_mask(state["latitudes"])
        state["longitudes"] = self._assign_mask(state["longitudes"])
        for field in state["fields"]:
            state["fields"][field] = self._assign_mask(state["fields"][field])

        return state

    def _assign_mask(self, array: np.ndarray):
        shape = array.shape[:-1] + self.mask.shape
        res = np.full(shape, self.fill_value, dtype=array.dtype)
        if array.ndim == 1:
            res[self.mask] = array
        else:
            res[..., self.mask] = array
        return res
