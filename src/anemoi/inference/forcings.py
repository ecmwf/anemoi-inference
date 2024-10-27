# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from abc import ABC
from abc import abstractmethod

import numpy as np

LOG = logging.getLogger(__name__)


class Forcings(ABC):
    """Represents the forcings for the model."""

    def __init__(self, runner):
        self.runner = runner

    @abstractmethod
    def load_forcings(self, state, date):
        pass


class ComputedForcings(Forcings):
    """Compute forcings like `cos_julian_day` or `insolation`."""

    def __init__(self, runner, variables, mask):
        super().__init__(runner)
        self.variables = variables
        self.mask = mask

    def load_forcings(self, state, date):
        LOG.debug("Adding dynamic forcings %s", self.variables)

        forcing = self.runner.compute_forcings(
            latitudes=state["latitudes"],
            longitudes=state["longitudes"],
            variables=self.variables,
            dates=[date],
        )

        return forcing.to_numpy(dtype=np.float32, flatten=True)


# TODO: Create a class `CoupledForcingsFromInput`
# That takes an Input object as a source
# CoupledForcingsFromInput(Mars)


class CoupledForcingsFromMars(Forcings):
    """Load forcings from Mars."""

    def __init__(self, runner, variables, mask):
        super().__init__(runner)
        self.variables = variables
        self.mask = mask
        self.grid = runner.checkpoint.grid
        self.area = runner.checkpoint.area
        self.use_grib_paramid = True  # TODO: find a way to `use_grib_paramid``

    def load_forcings(self, state, date):
        from .inputs.mars import retrieve

        requests = self.runner.checkpoint.mars_requests(
            date, use_grib_paramid=self.use_grib_paramid, variables=self.variables
        )

        fields = retrieve(requests=requests, grid=self.grid, area=self.area)

        warnings.warn("🚧 TEMPORARY CODE 🚧: Fields need to be sorted by name")

        return fields.to_numpy(dtype=np.float32, flatten=True)
