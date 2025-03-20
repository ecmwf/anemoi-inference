# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import List
from typing import Optional

from anemoi.inference.types import Date
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context

LOG = logging.getLogger(__name__)


class Input(ABC):
    """Abstract base class for input handling."""

    trace_name: str

    def __init__(self, context: "Context"):
        """Initialize the Input object.

        Parameters
        ----------
        context : Context
            The context for the input.
        """
        self.context = context
        self.checkpoint = context.checkpoint

    def __repr__(self) -> str:
        """Return a string representation of the Input object.

        Returns
        -------
        str
            The string representation of the Input object.
        """
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def create_state(self, *, date: Optional[Date], variables: Optional[List[str]], initial: bool) -> State:
        """Create the input state dictionary.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        variables : Optional[List[str]]
            The list of variables to include in the input state.
        initial : bool
            Whether the state is the initial state, in which case date expands to a list of dates
            according to the model's input time window lag.

        Returns
        -------
        State
            The input state dictionary.
        """
        pass

    def load_forcings_state(self, *, date: Date, variables: List[str], initial: bool) -> State:
        """Load forcings (constant and dynamic).

        Parameters
        ----------
        date : Date
            The date for which to load the forcings.
        variables : List[str]
            The list of variables to load.
        initial : bool
            Whether the state is the initial state, in which case date expands to a list of dates
            according to the model's input time window lag.

        Returns
        -------
        State
            The updated state with the loaded forcings.
        """
        return self.create_state(date=date, variables=variables, initial=initial)

    @property
    def checkpoint_variables(self) -> List[str]:
        """Return the list of input variables."""
        return list(self.checkpoint.variable_to_input_tensor_index.keys())

    def set_private_attributes(self, state: State, value: Any) -> None:
        """Provide a way to a subclass to set private attributes in the state
        dictionary, that may be needed by the output object.

        Parameters
        ----------
        state : State
            The state dictionary.
        value : Any
            The value to set.
        """
        pass
