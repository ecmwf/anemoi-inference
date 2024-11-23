# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd
import numpy as np

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("icon_grib_file")
class IconInput(GribInput):
    """
    Handles grib files from ICON
    WARNING: this code will become a pugin in the future
    """

    def __init__(self, context, path, grid, refinement_level_c, namer=None, **kwargs):
        super().__init__(context, namer=namer, **kwargs)
        self.path = path
        self.grid = grid
        self.refinement_level_c = refinement_level_c

    def create_input_state(self, *, date):
        import xarray as xr

        LOG.info(f"Reading ICON grid from {self.grid}")
        ds = xr.open_dataset(self.grid)
        latitudes = np.rad2deg(ds.clat[ds.refinement_level_c <= self.refinement_level_c].values)
        longitudes = np.rad2deg(ds.clon[ds.refinement_level_c <= self.refinement_level_c].values)

        LOG.info(f"Latitudes {np.min(latitudes)} {np.max(latitudes)}")
        LOG.info(f"Longitudes {np.min(longitudes)} {np.max(longitudes)}")

        LOG.info("Done")

        return self._create_input_state(
            ekd.from_source("file", self.path),
            variables=None,
            date=date,
            latitudes=latitudes,
            longitudes=longitudes,
        )

    def load_forcings(self, *, variables, dates):
        return self._load_forcings(ekd.from_source("file", self.path), variables=variables, dates=dates)
