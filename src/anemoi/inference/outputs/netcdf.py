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

import numpy as np

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("netcdf")
@main_argument("path")
class NetCDFOutput(Output):
    """_summary_"""

    def __init__(self, context, path, output_frequency=None, write_initial_state=None):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self.path = path
        self.ncfile = None
        self.float_size = "f4"

    def __repr__(self):
        return f"NetCDFOutput({self.path})"

    def __del__(self):
        if self.ncfile is not None:
            self.ncfile.close()

    def open(self, state):
        from netCDF4 import Dataset

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.ncfile = Dataset(self.path, "w", format="NETCDF4")

        compression = {}  # dict(zlib=False, complevel=0)

        values = len(state["latitudes"])

        self.values_dim = self.ncfile.createDimension("values", values)
        self.time_dim = self.ncfile.createDimension("time", size=None)  # infinite dimension
        self.time_var = self.ncfile.createVariable("time", "i8", ("time",), **compression)

        self.time_var.units = "seconds since {0}".format(self.reference_date)
        self.time_var.long_name = "time"
        self.time_var.calendar = "gregorian"

        latitudes = state["latitudes"]
        self.latitude_var = self.ncfile.createVariable(
            "latitude",
            self.float_size,
            ("values",),
            **compression,
        )
        self.latitude_var.units = "degrees_north"
        self.latitude_var.long_name = "latitude"

        longitudes = state["longitudes"]
        self.longitude_var = self.ncfile.createVariable(
            "longitude",
            self.float_size,
            ("values",),
            **compression,
        )
        self.longitude_var.units = "degrees_east"
        self.longitude_var.long_name = "longitude"

        self.vars = {}
        self.ensure_variables(state)

        self.latitude_var[:] = latitudes
        self.longitude_var[:] = longitudes

        self.n = 0
        return self.ncfile

    def ensure_variables(self, state):
        values = len(state["latitudes"])

        compression = {}  # dict(zlib=False, complevel=0)

        for name in state["fields"].keys():
            if name in self.vars:
                continue
            chunksizes = (1, values)

            while np.prod(chunksizes) > 1000000:
                chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            self.vars[name] = self.ncfile.createVariable(
                name,
                self.float_size,
                ("time", "values"),
                chunksizes=chunksizes,
                **compression,
            )

    def write_step(self, state):

        self.ensure_variables(state)

        self.time_var[self.n] = self.step(state).total_seconds()

        for name, value in state["fields"].items():
            self.vars[name][self.n] = value

        self.n += 1

    def close(self):
        if self.ncfile is not None:
            self.ncfile.close()
            self.ncfile = None
