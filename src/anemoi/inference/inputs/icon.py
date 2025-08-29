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

import earthkit.data as ekd

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("icon_grib_file")
class IconInput(GribInput):
    """Handles grib files from ICON."""

    # TODO: this code will become a plugin in the future.

    trace_name = "icon file"

    def __init__(
        self,
        context: Context,
        path: str,
        grid: str,
        refinement_level_c: int,
        pre_processors: list[ProcessorConfig] | None = None,
        namer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the IconInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        path : str
            The path to the ICON grib file.
        grid : str
            The grid type.
        refinement_level_c : int
            The refinement level.
        pre_processors : Optional[List[ProcessorConfig]], default None
            Pre-processors to apply to the input
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, pre_processors, namer=namer, **kwargs)
        self.path = path
        self.grid = grid
        self.refinement_level_c = refinement_level_c

    def create_input_state(self, *, date: Date | None) -> State:
        """Creates the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The created input state.
        """
        from anemoi.transform.grids.icon import icon_grid

        latitudes, longitudes = icon_grid(self.grid, self.refinement_level_c)

        return self._create_input_state(
            ekd.from_source("file", self.path),
            variables=None,
            date=date,
            latitudes=latitudes,
            longitudes=longitudes,
        )

    def load_forcings_state(self, *, variables: list[str], dates: list[Date], current_state: State) -> State:
        """Loads the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            List of variable names.
        dates : List[Date]
            List of dates for which to load the forcings state.
        current_state : State
            The current state.

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
