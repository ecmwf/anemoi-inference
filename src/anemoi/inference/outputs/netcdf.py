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
import os
import threading
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig, State

from ..decorators import ensure_path, main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


@output_registry.register("netcdf")
@main_argument("path")
@ensure_path("path")
class NetCDFOutput(Output):
    """NetCDF output class."""

    def __init__(
        self,
        context: Context,
        path: Path,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        float_size: str = "f4",
        missing_value: float | None = np.nan,
    ) -> None:
        """Initialise the NetCDF output object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        path : Path
            The path to save the NetCDF file to.
            If the parent directory does not exist, it will be created.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        float_size : str, optional
            The size of the float, by default "f4".
        missing_value : float, optional
            The missing value, by default np.nan.
        """

        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )

        from netCDF4 import Dataset

        self.path = path
        self.ncfile: Dataset | None = None
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

        with LOCK:
            if self.ncfile is not None:
                return

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        # Try to get template path from context
        template_path = getattr(self.context, "output_template", None) or getattr(
            getattr(self.context, "development_hacks", {}), "output_template", None
        )

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

        with LOCK:
            self.ncfile = Dataset(self.path, "w", format="NETCDF4")

        state = self.post_process(state)
        compression = {}  # dict(zlib=False, complevel=0)

        time = 0
        self.reference_date = state["date"]
        if (time_step := getattr(self.context, "time_step", None)) and (
            lead_time := getattr(self.context, "lead_time", None)
        ):
            time = lead_time // time_step
            time += self.extra_time

        if reference_date := getattr(self.context, "reference_date", None):
            self.reference_date = reference_date

        with LOCK:
            self.values_dim = self.ncfile.createDimension("values", values)
            self.time_dim = self.ncfile.createDimension("time", time)
            self.time_var = self.ncfile.createVariable(
                "time", "i4", ("time",), **compression
            )

            self.time_var.units = f"seconds since {self.reference_date}"
            self.time_var.long_name = "time"
            self.time_var.calendar = "gregorian"

        with LOCK:
            self.latitude_var = self.ncfile.createVariable(
                "latitude", self.float_size, ("values",), **compression
            )
            self.latitude_var.units = "degrees_north"
            self.latitude_var.long_name = "latitude"

            self.longitude_var = self.ncfile.createVariable(
                "longitude", self.float_size, ("values",), **compression
            )
            self.longitude_var.units = "degrees_east"
            self.longitude_var.long_name = "longitude"

            self.latitude_var[:] = latitudes
            self.longitude_var[:] = longitudes

        self.n = 0

        LOG.info(f"🕒 Reference date set to {self.reference_date}")

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

            if self.skip_variable(name):
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
            if self.skip_variable(name):
                continue

            with LOCK:
                LOG.debug(f"🚧🚧🚧🚧🚧🚧 XXXXXX {name}, {self.n}, {value.shape}")
                self.vars[name][self.n] = value

        self.n += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
