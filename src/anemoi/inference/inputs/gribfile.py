# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from functools import cached_property
from typing import Any

import earthkit.data as ekd

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..decorators import main_argument
from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("grib")
@main_argument("path")
class GribFileInput(GribInput):
    """Handles grib files."""

    trace_name = "grib file"

    def __init__(
        self,
        context: Context,
        *,
        path: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the GribFileInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        path : str
            The path to the GRIB file.
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)
        self.path = path

    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        **kwargs : Any
            Additional keyword arguments, including:
            - ref_date_index: int, default -1
                The reference date index to use.

        Returns
        -------
        State
            The created input state.
        """
        ref_date_index = kwargs.get("ref_date_index", -1)
        return self._create_input_state(self._fieldlist, date=date, ref_date_index=ref_date_index)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        dates : List[Date]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        State
            The loaded forcings state.
        """

        return self._load_forcings_state(
            self._fieldlist,
            dates=dates,
            current_state=current_state,
        )

    @cached_property
    def _fieldlist(self) -> ekd.FieldList:
        """Get the input fieldlist from the GRIB file."""
        if os.path.getsize(self.path) == 0:
            LOG.warning("GRIB file %r is empty", self.path)
            return ekd.from_source("empty")
        return ekd.from_source("file", self.path)
