# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
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

# TODO: only one method is need: `load_data`.
# The other methods can be implemenneted concreetly
# using the `load_data` method.


class Input(ABC):
    """Abstract base class for input handling."""

    trace_name = "????"  # Override in subclass

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
    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create the input state dictionary.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The input state dictionary.
        """
        pass

    @abstractmethod
    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        """Load forcings (constant and dynamic).

        Parameters
        ----------
        variables : List[str]
            The list of variables to load.
        dates : List[Date]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        State
            The updated state with the loaded forcings.
        """
        pass

    def input_variables(self) -> List[str]:
        """Return the list of input variables.

        Returns
        -------
        List[str]
            The list of input variables.
        """
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
