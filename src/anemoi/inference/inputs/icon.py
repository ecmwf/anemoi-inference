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
from anemoi.transform.grids.icon import icon_grid

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("icon_grib_file")
class IconInput(GribInput):
    """Handles grib files from ICON
    WARNING: this code will become a pugin in the future.
    """

    trace_name = "icon file"

    def __init__(
        self, context: Any, path: str, grid: str, refinement_level_c: int, namer: Optional[Any] = None, **kwargs: Any
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
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, namer=namer, **kwargs)
        self.path = path
        self.grid = grid
        self.refinement_level_c = refinement_level_c

    def create_input_state(self, *, date: Optional[Any]) -> Any:
        latitudes, longitudes = icon_grid(self.grid, self.refinement_level_c)

        return self._create_state(
            ekd.from_source("file", self.path),
            variables=None,
            date=date,
            latitudes=latitudes,
            longitudes=longitudes,
        )

    def load_forcings_state(self, *, variables: List[str], dates: List[Any], current_state: Any) -> Any:
        return self._load_forcings_state(
            ekd.from_source("file", self.path), variables=variables, dates=dates, current_state=current_state
        )
