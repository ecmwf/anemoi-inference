# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
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

    def __init__(self, context):
        self.context = context
        self.checkpoint = context.checkpoint
        self.kinds = dict(unknown=True)  # Used for debugging

    @abstractmethod
    def load_forcings(self, state, date):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ComputedForcings(Forcings):
    """Compute forcings like `cos_julian_day` or `insolation`."""

    def __init__(self, context, variables, mask):
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(computed=True)  # Used for debugging

    def load_forcings(self, state, dates):
        LOG.debug("Adding dynamic forcings %s", self.variables)

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        forcing = self.context.compute_forcings(
            latitudes=state["latitudes"],
            longitudes=state["longitudes"],
            variables=self.variables,
            dates=dates,
        )

        # Forcing are sorted by `compute_forcings`  in the order (varaible, date)

        return forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(self.variables), len(dates), -1)


class CoupledForcings(Forcings):
    """Retrieve forcings from the input."""

    def __init__(self, context, input, variables, mask):
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.grid = context.checkpoint.grid
        self.area = context.checkpoint.area
        self.use_grib_paramid = True  # TODO: find a way to `use_grib_paramid``
        self.kinds = dict(retrieved=True)  # Used for debugging

    def load_forcings(self, state, dates):
        from .inputs.mars import retrieve

        requests = self.context.checkpoint.mars_requests(
            variables=self.variables,
            dates=dates,
            use_grib_paramid=self.use_grib_paramid,
        )

        if not requests:
            raise ValueError("No requests for %s (%s)" % (self.variables, dates))

        for r in requests:
            LOG.info("Request: %s", r)

        fields = retrieve(requests=requests, grid=self.grid, area=self.area, expver=1)

        if not fields:
            raise ValueError("No fields retrieved for {self.variables} ({dates})")

        fields = self.checkpoint.name_fields(fields).order_by(name=self.variables, valid_datetime="ascending")

        return fields.to_numpy(dtype=np.float32, flatten=True).reshape(len(self.variables), len(dates), -1)
