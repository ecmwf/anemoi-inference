# (C) Copyright 2024-2026 Anemoi contributors.
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
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import ensure_path
from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..decorators import supports_parallel_output
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()

CALENDAR = "standard"


@output_registry.register("netcdf")
@main_argument("path")
@format_dataset_name("path")
@supports_parallel_output("path")
@ensure_path("path")
class NetCDFOutput(Output):
    """NetCDF output class."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
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
        variables : list, optional
            The list of variables to write, by default None.
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
            metadata,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )

        from netCDF4 import Dataset

        self.ncfile: Dataset | None = None
        self.float_size = float_size
        self.missing_value = missing_value
        self.path = path

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

        state = self.post_process(state)

        compression = {}  # dict(zlib=False, complevel=0)

        n_values = len(state["latitudes"])

        # set epoch
        epoch = getattr(self.context, "reference_date", None) or state["date"]
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        self.epoch = epoch

        # start date of the forecast
        self.reference_date = np.int64(_to_epoch_seconds(state["date"], self.epoch))

        # Dimensions
        with LOCK:
            self.time_dim = self.ncfile.createDimension("time", None)
            self.values_dim = self.ncfile.createDimension("values", n_values)

        # start time of forecast
        # no dimensions, scalar variable
        with LOCK:
            self.reference_date_var = self.ncfile.createVariable("forecast_reference_time", "i8")
            self.reference_date_var.standard_name = "forecast_reference_time"
            self.reference_date_var.long_name = "start time of forecast"
            self.reference_date_var.units = f"seconds since {self.epoch}"
            self.reference_date_var.calender = CALENDAR
            self.reference_date_var[:] = self.reference_date

        # valid time
        # time dimension coordinate
        with LOCK:
            self.time_var = self.ncfile.createVariable("time", "i8", ("time",), **compression)
            self.time_var.standard_name = "time"
            self.time_var.long_name = "valid time"
            self.time_var.units = f"seconds since {self.epoch}"
            self.time_var.calendar = CALENDAR
            self.time_var.axis = "T"

        # forecast period / lead time
        # time dimension auxilary coordinate
        with LOCK:
            self.period_var = self.ncfile.createVariable("forecast_period", "i8", ("time",), **compression)
            self.period_var.standard_name = "forecast_period"
            self.period_var.long_name = "lead time"
            self.period_var.units = "seconds"

        # latitude / longitude
        # values dimension auxilary coordinates
        with LOCK:
            latitudes = state["latitudes"]
            self.lat_var = self.ncfile.createVariable("latitude", self.float_size, ("values",), **compression)
            self.lat_var.standard_name = "latitude"
            self.lat_var.units = "degrees_north"
            self.lat_var[:] = latitudes

            longitudes = state["longitudes"]
            self.lon_var = self.ncfile.createVariable("longitude", self.float_size, ("values",), **compression)
            self.lon_var.standard_name = "longitude"
            self.lon_var.units = "degrees_east"
            self.lon_var[:] = longitudes

        self.n = 0
        self.vars = {}

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
            if self.skip_variable(name):
                continue

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
                self.vars[name].coordinates = "forecast_reference_time forecast_period latitude longitude"

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        self.ensure_variables(state)

        step = np.int64(_to_epoch_seconds(state["date"], self.epoch)) - self.reference_date

        # update time coordinates
        self.period_var[self.n] = step
        self.time_var[self.n] = self.reference_date + step

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


def _to_epoch_seconds(dt: datetime, epoch: datetime) -> int:
    """Exact integer seconds since epoch, from a plain python datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # assume naive datetimes are UTC
    delta = dt - epoch
    return delta.days * 86400 + delta.seconds
