# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
from typing import Optional

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..output import ForwardOutput

LOG = logging.getLogger(__name__)


class MaskedOutput(ForwardOutput):
    """Apply mask output class.

    Parameters
    ----------
    context : dict
        The context dictionary.
    mask : Any
        The mask.
    output : dict
        The output configuration dictionary.
    output_frequency : int, optional
        The frequency of output, by default None.
    write_initial_state : bool, optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self,
        context: Context,
        *,
        mask: Any,
        output: Configuration,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        super().__init__(context, output, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self.mask = mask

    def modify_state(self, state: State) -> State:
        """Apply the mask to the state.

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

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}({self.mask}, {self.output})"
