# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Dummy input used for testing.

It will generate fields with constant values for each variable and date.
These values are then tested in the mock model.
"""

import logging
from typing import Any
from typing import List
from typing import Optional

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)
SKIP_KEYS = ["date", "time", "step"]


@input_registry.register("empty")
class EmptyInput(Input):
    """An Input that is always empty."""

    trace_name = "empty"

    def __init__(self, context: Context, **kwargs: Any) -> None:
        """Initialise the EmptyInput.

        Parameters
        ----------
        context : Context
            The context object for the input.
        **kwargs : object
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)
        assert self.variables in (None, []), "EmptyInput should not have variables"

    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create an empty input state.

        Parameters
        ----------
        date : Date or None
            The date for the input state.

        Returns
        -------
        State
            The created empty input state.
        """
        return dict(fields=dict(), _input=self)

    def load_forcings_state(self, *, dates: List[Date], current_state: State) -> State:
        """Load an empty forcings state.

        Parameters
        ----------
        dates : list of Date
            The list of dates for the forcings state.
        current_state : State
            The current state (unused).

        Returns
        -------
        State
            The loaded empty forcings state.
        """
        return dict(date=dates[-1], fields=dict(), _input=self)
