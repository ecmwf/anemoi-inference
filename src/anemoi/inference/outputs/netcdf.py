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
from typing import Any, Optional

import numpy as np
import pyproj
from netCDF4 import Dataset, Variable

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig, State

from ..decorators import ensure_path, main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


class FieldVar:
    def __init__(self, name: str, attrs: dict[str, Any], projected: bool) -> None:
        self.name = name
        self.attrs = attrs
        if projected:
            self.attrs["grid_mapping"] = "projection"
            self.attrs["coordinate"] = "latitude longitude"


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

        self.path = path
        self.float_size = float_size
        self.missing_value = missing_value
        self.compression = {}  # dict(zlib=False, complevel=0)

        if self.write_step_zero:
            self.extra_time = 1
        else:
            self.extra_time = 0

        self.template = None  # Add this one line so open() won't crash

        # timestep number
        self.n = 0
        self.proj_str = getattr(self.context, "projection_string", None)

        # TODO: is something like this available somewhere inside state?
        # Otherwise we probably need to have it as input in the config?
        self.variable_metadata = {
            "2t": FieldVar(
                name="air_temperature_2m",
                attrs={"units": "K", "standard_name": "air_temperature"},
                projected=self.proj_str is not None,
            )
        }

        # TODO: to be fair this doesn't look that good?
        # Why can't we have __init__ actually open the file?
        self.ncfile: Optional[Dataset] = None
        self.field_shape: tuple[int, ...]
        # dimesions for the field variables
        self.dimesions: tuple[str, ...]
        self.reference_date: datetime.datetime
        self.time: Variable
        self.vars: dict[str, Variable]

    def _set_reference_date(self, state: State):
        # TODO: this should be "reference_date" but it's not implemented?
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
        LOG.info(f"Reference date set to {ref_date}")

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def _create_var(
        self,
        name: str,
        dtype: str,
        dims: tuple[str, ...],
        units: str,
        values: Optional[np.ndarray] = None,
    ) -> Variable:
        assert self.ncfile is not None

        var = self.ncfile.createVariable(name, dtype, dims)
        var.units = units
        if values is not None:
            var[:] = values

        return var

    def _create_field_var(self, name: str) -> Variable:
        """Create a variable with missing values"""
        assert self.ncfile is not None

        if name in self.variable_metadata:
            metadata = self.variable_metadata[name]
            outname = metadata.name
            attrs = metadata.attrs
        else:
            outname = name
            attrs = {}

        var = self.ncfile.createVariable(
            outname,
            self.float_size,
            self.dimesions,
            fill_value=self.missing_value,
            **self.compression,
        )

        var.missing_value = self.missing_value
        var.fill_value = self.missing_value

        var.setncatts(attrs)

        return var

    def open(self, state: State) -> None:
        """Open the NetCDF file and initialize dimensions and variables.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        with LOCK:
            if self.ncfile is not None:
                return

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        self._set_reference_date(state)

        state = self.post_process(state)

        time = None
        self.reference_date = state["date"]
        if (time_step := getattr(self.context, "time_step", None)) and (
            lead_time := getattr(self.context, "lead_time", None)
        ):
            time = lead_time // time_step
            time += self.extra_time

        # TODO: provide these by template or via config file
        self.field_shape = (989, 789)
        (y_size, x_size) = self.field_shape

        # TODO: also keep previous dimensions? (time, values)?
        self.dimesions = ("time", "height", "y", "x")

        template_path = getattr(
            getattr(self.context, "development_hacks", {}), "output_template"
        )

        if template_path is not None:
            with Dataset(template_path, "r") as template:
                # x_size = len(template.dimensions["x"])
                # y_size = len(template.dimensions["x"])
                # self.field_shape = (y_size, x_size)

                lat_values = template.variables["latitude"][:]
                lon_values = template.variables["longitude"][:]
        else:
            # TODO: fix this path
            lat_values = state["latitudes"]
            lon_values = state["longitudes"]
            n_values = len(lat_values)
            self.dimesions = ("time", "height", "values")

        # TODO: these don't have the correct shape if loaded from state?
        lats = np.reshape(lat_values, self.field_shape)
        lons = np.reshape(lon_values, self.field_shape)

        # Create new NetCDF file at self.path
        with LOCK:
            if self.ncfile is not None:
                return

            self.ncfile = Dataset(self.path, "w", format="NETCDF4")
            self.ncfile.createDimension("time", time)
            self.ncfile.createDimension("y", y_size)
            self.ncfile.createDimension("x", x_size)
            self.ncfile.createDimension("height", 1)

            # TODO: i4 or f8 for time?
            self.time = self._create_var(
                "time", "i4", ("time",), "seconds since 1970-01-01T00:00:00Z"
            )

            self._create_var("latitude", "f8", self.dimesions, "degrees_north", lats)
            self._create_var("longitude", "f8", self.dimesions, "degrees_east", lons)

            if self.proj_str is not None:
                x, y = self._get_projections(lats, lons, self.proj_str)

                self._create_var("x", "f4", ("x",), "m", x)
                self._create_var("y", "f4", ("y",), "m", y)
                self._create_projection_var(self.proj_str)

        LOG.info(
            f"Created NetCDF file {self.path} with dimensions: "
            f"(time=unlimited, height=1, y={y_size}, x={x_size})"
        )

    def _create_projection_var(self, proj_str: str):
        assert self.ncfile is not None

        var = self.ncfile.createVariable("projection", "i4", [])

        # Convert projection string to dictionary of attributes
        crs = pyproj.CRS.from_proj4(proj_str)

        # Set those attributes to the projection variable
        attrs = {
            k: v for k, v in crs.to_cf().items() if v != "unknown" or k != "crs_wkt"
        }
        attrs["earth_radius"] = 6_371_000.0

        var.setncatts(attrs)

    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        assert isinstance(self.ncfile, Dataset), "netcdf file not initialized"

        for name in state["fields"].keys():
            if name in self.vars:
                continue

            if self.skip_variable(name):
                continue

            # TODO: actually chunk? Not used right now
            # chunksizes = (1, min(values, 100000))
            #
            # while np.prod(chunksizes) > 1000000:
            #     chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            with LOCK:
                self.vars[name] = self._create_field_var(name)

    @staticmethod
    def _get_projections(
        lats: np.ndarray, lons: np.ndarray, proj_str: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reverse engineer x and y vectors from lats and lons"""

        proj_from = pyproj.Proj("proj+=longlat")
        proj_to = pyproj.Proj(proj_str)

        transformer = pyproj.Transformer.from_proj(proj_from, proj_to)

        xx, yy = transformer.transform(lons, lats)
        x = xx[0, :]
        y = yy[:, 0]

        return x, y

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        # TODO: why are these not initialized inside open()?
        self.ensure_variables(state)

        # Store absolute time since 1970 (not relative to reference_date)
        epoch = datetime.datetime(1970, 1, 1)
        step = state["date"] - epoch

        self.time[self.n] = step.total_seconds()

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            with LOCK:
                LOG.debug(f"🚧🚧🚧🚧🚧🚧 Writing {name}, {self.n}, {self.field_shape}")

                field_2d = np.reshape(value, self.field_shape)

                # TODO: abstract away the slice object?
                self.vars[name][self.n, 0, :, :] = field_2d

        self.n += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
