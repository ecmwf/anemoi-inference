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
import datetime

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
    """NetCDF output class."""

    def __init__(
        self,
        context: Context,
        path: str,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
        float_size: str = "f4",
        missing_value: Optional[float] = np.nan,
    ) -> None:
        """Initialize the NetCDF output object.

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
        float_size : str, optional
            The size of the float, by default "f4".
        missing_value : float, optional
            The missing value, by default np.nan.
        """

        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        from netCDF4 import Dataset

        self.path = path
        self.ncfile: Optional[Dataset] = None
        self.float_size = float_size
        self.missing_value = missing_value
        if self.write_step_zero:
            self.extra_time = 1
        else:
            self.extra_time = 0
        self.vars = {}
        self.template = None  # Add this one line so open() won't crash

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
        import datetime

        # Try to get template path from context
        template_path = getattr(self.context, "output_template", None) or \
                        getattr(getattr(self.context, "development_hacks", {}), "output_template", None)

        if template_path is not None and os.path.exists(template_path):
            LOG.info(f"📄 Using output template: {template_path}")
            with Dataset(template_path) as tmpl:
                values = len(tmpl.dimensions["values"])
                latitudes = tmpl.variables["latitude"][:]
                longitudes = tmpl.variables["longitude"][:]
        else:
            LOG.info("⚠️ No template provided, falling back to state lat/lon")
            values = len(state["latitudes"])
            latitudes = state["latitudes"]
            longitudes = state["longitudes"]

        LOG.info(f"📏 NetCDFOutput.open: values={values}")

        # Create new NetCDF file at self.path
        self.ncfile = Dataset(self.path, "w", format="NETCDF4")
        self.ncfile.createDimension("time", None)  # unlimited
        self.ncfile.createDimension("values", values)

        self.time_var = self.ncfile.createVariable("time", "f8", ("time",))
        self.lat_var = self.ncfile.createVariable("latitude", "f8", ("values",))
        self.lon_var = self.ncfile.createVariable("longitude", "f8", ("values",))

        self.time_var.units = "seconds since 1970-01-01 00:00:00"
        self.lat_var.units = "degrees_north"
        self.lon_var.units = "degrees_east"

        self.lat_var[:] = latitudes
        self.lon_var[:] = longitudes

        self.n = 0
        
        # Reference date
        ref_date = getattr(self.context, "date", None)
        
        if ref_date is None:
            dates_obj = getattr(self.context, "dates", None)
            if dates_obj is not None:
                ref_date = getattr(dates_obj, "start", None)

        if isinstance(ref_date, str):
            ref_date = datetime.datetime.fromisoformat(ref_date)

        if ref_date is None:
            # fallback: use state date
            ref_date = state["date"]

        self.reference_date = ref_date

        LOG.info(f"🕒 Reference date set to {self.reference_date}")
        LOG.info(f"✅ Created NetCDF file {self.path} with dimensions: time=unlimited, values={values}")

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
                missing_value = self.missing_value

                self.vars[name] = self.ncfile.createVariable(
                    name,
                    self.float_size,
                    ("time", "values"),
                    chunksizes=chunksizes,
                    fill_value=missing_value,
                    **compression,
                )

                self.vars[name].fill_value = missing_value
                self.vars[name].missing_value = missing_value
                
    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        # Use the dimension already defined in open()
        values = len(self.ncfile.dimensions["values"])
        compression = {}  # dict(zlib=False, complevel=0)

        for name in state["fields"].keys():
            if name in self.vars:
                continue
            chunksizes = (1, min(values, 100000))
            
            while np.prod(chunksizes) > 1000000:
                chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            with LOCK:
                missing_value = self.missing_value

                self.vars[name] = self.ncfile.createVariable(
                    name,
                    self.float_size,
                    ("time", "values"),
                    fill_value=missing_value,
                    **compression,
                )

                self.vars[name].fill_value = missing_value
                self.vars[name].missing_value = missing_value

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        self.ensure_variables(state)

        # Store absolute time since 1970 (not relative to reference_date)
        epoch = datetime.datetime(1970, 1, 1)
        step = state["date"] - epoch
        self.time_var[self.n] = step.total_seconds()

        for name, value in state["fields"].items():
            with LOCK:
                LOG.info(f"🚧🚧🚧🚧🚧🚧 XXXXXX {name}, {self.n}, {value.shape}")
                self.vars[name][self.n] = value

        self.n += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
