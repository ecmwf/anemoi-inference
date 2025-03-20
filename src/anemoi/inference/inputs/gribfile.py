# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any
from typing import Callable
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
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
        path: str,
        *,
        namer: Optional[Callable[[Any, Any], str]] = None,
    ) -> None:
        """Initialize the GribInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        path : str
            The path to the GRIB file.
        namer : Optional[Any]
            Optional namer for the input.
        """
        super().__init__(context, namer)
        self.path = path

    # TODO: we might also implement the file-pattern source
    def _earthkit_reader(self, path):
        return ekd.from_source(path)


@input_registry.register("icon_grib_file")
class IconInput(GribFileInput):
    """Handles grib files from ICON."""

    # TODO: this code will become a plugin in the future.

    trace_name = "icon file"

    def __init__(
        self,
        context: Context,
        path: str,
        grid: str,
        refinement_level_c: int,
        namer: Optional[Any] = None,
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
        namer : Optional[Any]
            Optional namer for the input.
        """
        super().__init__(context, path, namer=namer)
        self.grid = grid
        self.refinement_level_c = refinement_level_c

    def fieldlist_to_state(self, fieldlist: ekd.FieldList) -> State:
        """Convert a fieldlist to a state dictionary.

        Parameters
        ----------
        fieldlist : earthkit.data.FieldList
            The fieldlist to convert.

        Returns
        -------
        State
            The converted state.
        """
        from anemoi.transform.grids.icon import icon_grid

        state = {"fields": {}}
        for field in fieldlist.group_by("name"):
            name = field.metadata("name")
            state["fields"][name] = field.values

        date = fieldlist.metadata("valid_datetime")[-1]
        state["date"] = np.datetime64(date).astype(datetime.datetime)

        latitudes, longitudes = icon_grid(self.grid, self.refinement_level_c)
        state["latitudes"] = latitudes
        state["longitudes"] = longitudes

        return state
