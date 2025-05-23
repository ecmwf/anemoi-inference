# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import List
from typing import Optional

import numpy as np

from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry

LOG = logging.getLogger(__name__)


def _nest_state(
    state: State,
    _state: State,
) -> State:
    """Nest the state with the given mask.

    Parameters
    ----------
    state : State
        The state to be nested.
    _state : State
        The state to be nested with.

    Returns
    -------
    State
        The nested state.
    """
    for field, values in state["fields"].items():
        state["fields"][field] = np.concatenate([values, _state["fields"][field]], axis=-1)

    state["latitudes"] = np.concatenate([state["latitudes"], _state["latitudes"]], axis=-1)
    state["longitudes"] = np.concatenate([state["longitudes"], _state["longitudes"]], axis=-1)

    return state


@input_registry.register("nest")
class NestedInputs(Input):
    """Combines two or more input sources with nesting.

    This applies no masking, so use `cutout` if you want to mask the input.

    Example
    -------
    From config:

    ```yaml
    nest:
        inputs:
            - mars:
                area: 2/-2/-2/2
                grid: 0.01/0.01
            - mars
    ```
    This will create a nested input from the two sources.
    """

    def __init__(self, context, inputs: List[dict | Input]):
        """Create a nested input from a list of sources.

        Parameters
        ----------
        context : dict
            The context runner.
        inputs :  List[dict] of sources
            A list of dictionaries of sources to combine.
        """
        super().__init__(context)

        self.inputs = [create_input(context, cfg) if not isinstance(cfg, Input) else cfg for cfg in inputs]

    def __repr__(self):
        return f"Nested({self.inputs})"

    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The created input state.
        """

        LOG.info(f"Concatenating states from {self.inputs}")

        state = self.inputs[0].create_input_state(date=date)

        for input_source in self.inputs[1:]:
            _state = input_source.create_input_state(date=date)

            state = _nest_state(state, _state)

        return state

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            List of variables to load.
        dates : List[Date]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        State
            The loaded forcings state.
        """

        state = self.inputs[0].load_forcings_state(variables=variables, dates=dates, current_state=current_state)
        fields = state["fields"]

        for input_source in self.inputs[1:]:
            _state = input_source.load_forcings_state(variables=variables, dates=dates, current_state=current_state)

            state = _nest_state(state, _state)

        current_state["fields"] |= fields
        return current_state
