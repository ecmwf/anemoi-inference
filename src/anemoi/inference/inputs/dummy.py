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
from typing import List
from typing import Optional

import earthkit.data as ekd

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..decorators import main_argument
from . import input_registry
from .grib import GribInput
from .ekd import EkdInput


LOG = logging.getLogger(__name__)

@input_registry.register("dummy")
class DummyInput(EkdInput):
    """Dummy input used for testing."""


    trace_name = "dummy"

    def __init__(self, context: Context, *, namer: Optional[Any] = None, **kwargs: Any) -> None:
        """Initialize the DummyInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, namer=namer, **kwargs)

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
        return self._create_state(ekd.from_source("file", self.path), variables=None, date=date)

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
        Any
            The loaded forcings state.
        """
        return self._load_forcings_state(
            ekd.from_source("file", self.path),
            variables=variables,
            dates=dates,
            current_state=current_state,
        )

    def _fields(self, date: Optional[Date], variables: Optional[List[str]]):
        if variables is None:
            variables = self.checkpoint.variables_from_input(include_forcings=True)

        assert False, variables
