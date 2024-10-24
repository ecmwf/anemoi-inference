# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import numpy as np

LOG = logging.getLogger(__name__)


class Forcings(ABC):
    """Represents the forcings for the model."""

    def __init__(self, runner, *, verbose: bool = True):
        self.runner = runner
        self._verbose = verbose

    @abstractmethod
    def load_forcings(self, state, date):
        pass


class ComputedForcings(Forcings):
    """Compute forcings like `cos_julian_day` or `insolation`."""

    def __init__(self, runner, variables, mask, *, verbose: bool = True):
        super().__init__(runner, verbose=verbose)
        self.variables = variables
        self.mask = mask

    def load_forcings(self, state, date):
        LOG.info("Adding dynamic forcings %s", self.variables)

        forcing = self.runner.compute_forcings(
            latitudes=state["latitudes"],
            longitudes=state["longitudes"],
            variables=self.variables,
            dates=[date],
        )

        return forcing.to_numpy(dtype=np.float32, flatten=True)
