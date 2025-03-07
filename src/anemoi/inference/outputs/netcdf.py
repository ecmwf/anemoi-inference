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
import threading
from typing import Optional

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


@output_registry.register("netcdf")
@main_argument("path")
class NetCDFOutput(Output):
    """NetCDF output class.

    Parameters
    ----------
    context : dict
        The context dictionary.
    path : str
        The path to save the NetCDF file.
    output_frequency : int, optional
        The frequency of output, by default None.
    write_initial_state : bool, optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self,
        context: Context,
        path: str,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        from netCDF4 import Dataset

        self.path = path
        self.ncfile: Optional[Dataset] = None
        self.float_size = "f4"

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def open(self, state: State) -> None:
        """Open the NetCDF file and initialize dimensions and variables.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        from netCDF4 import Dataset

        with LOCK:
            if self.ncfile is not None:
                return

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        with LOCK:
            self.ncfile = Dataset(self.path, "w", format="NETCDF4")

        compression = {}  # dict(zlib=False, complevel=0)

        values = len(state["latitudes"])

        time = 0
        self.reference_date = state["date"]
        if hasattr(self.context, "time_step") and hasattr(self.context, "lead_time"):
            time = self.context.lead_time // self.context.time_step
        if hasattr(self.context, "reference_date"):
            self.reference_date = self.context.reference_date

        with LOCK:
            self.values_dim = self.ncfile.createDimension("values", values)
            self.time_dim = self.ncfile.createDimension("time", time)
            self.time_var = self.ncfile.createVariable("time", "i4", ("time",), **compression)

            self.time_var.units = "seconds since {0}".format(self.reference_date)
            self.time_var.long_name = "time"
            self.time_var.calendar = "gregorian"

        latitudes = state["latitudes"]
        with LOCK:
            self.latitude_var = self.ncfile.createVariable("latitude", self.float_size, ("values",), **compression)
            self.latitude_var.units = "degrees_north"
            self.latitude_var.long_name = "latitude"

        longitudes = state["longitudes"]
        with LOCK:
            self.longitude_var = self.ncfile.createVariable("longitude", self.float_size, ("values",), **compression)
            self.longitude_var.units = "degrees_east"
            self.longitude_var.long_name = "longitude"

        self.latitude_var[:] = latitudes
        self.longitude_var[:] = longitudes

        self.vars = {}
        self.ensure_variables(state)

        self.n = 0

    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        values = len(state["latitudes"])

        compression = {}  # dict(zlib=False, complevel=0)

        for name in state["fields"].keys():
            if name in self.vars:
                continue
            chunksizes = (1, values)

            while np.prod(chunksizes) > 1000000:
                chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            with LOCK:
                self.vars[name] = self.ncfile.createVariable(
                    name,
                    self.float_size,
                    ("time", "values"),
                    chunksizes=chunksizes,
                    **compression,
                )

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        self.ensure_variables(state)

        step = state["date"] - self.reference_date
        self.time_var[self.n] = step.total_seconds()

        for name, value in state["fields"].items():
            with LOCK:
                self.vars[name][self.n] = value

        self.n += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
