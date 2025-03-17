# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..output import ForwardOutput
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("apply_mask")
class ApplyMaskOutput(ForwardOutput):
    """Apply mask output class.

    Parameters
    ----------
    context : dict
        The context dictionary.
    mask : str
        The mask identifier.
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
        mask: str,
        output: Configuration,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self.mask = self.checkpoint.load_supporting_array(mask)
        self.output = create_output(context, output)

    def __repr__(self) -> str:
        """Return a string representation of the ApplyMaskOutput object."""
        return f"ApplyMaskOutput({self.mask}, {self.output})"

    def write_initial_step(self, state: State) -> None:
        """Write the initial step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        # Note: we foreward to 'state', so we write-up options again
        self.output.write_initial_state(self._apply_mask(state))

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        # Note: we foreward to 'state', so we write-up options again
        self.output.write_state(self._apply_mask(state))

    def _apply_mask(self, state: State) -> State:
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

    def close(self) -> None:
        """Close the output."""
        self.output.close()

    def print_summary(self, depth: int = 0) -> None:
        """Print the summary of the output.

        Parameters
        ----------
        depth : int, optional
            The depth of the summary, by default 0.
        """
        super().print_summary(depth)
        self.output.print_summary(depth + 1)
