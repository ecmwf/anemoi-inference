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
import xarray as xr
from netCDF4 import Dataset, Variable

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig, State

from ..decorators import ensure_path, main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


class VarMetadata:
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.attrs = attrs


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
        self.proj_str: Optional[str] = getattr(
            self.context.development_hacks, "projection_string", None
        )
        self.members = getattr(self.context.development_hacks, "n_members", 1)

        # TODO: is something like this available somewhere inside state?
        # Otherwise we probably need to have it as input in the config?
        self.variable_metadata = {
            "2t": VarMetadata(
                name="air_temperature_2m",
                attrs={"units": "K", "standard_name": "air_temperature"},
            )
        }

        self.ncfile: Optional[Dataset] = None
        self.vars: dict[str, Variable] = {}

        # shape of the fields
        self.field_shape: tuple[int, ...]

        # netcdf dimesions for the field variables
        self.dimensions: tuple[str, ...]

        # Reference date for the time axis
        # TODO: self.reference_date: datetime.datetime = datetime.datetime(1970, 1, 1)
        self.reference_date: datetime.datetime

        # netcdf time variable
        self.time: Variable

    def _get_reference_date(self, state: State) -> datetime.datetime:
        # TODO: this should be "reference_date" in the base context but it's not implemented?
        ref_date = getattr(self.context.development_hacks, "reference_date")

        if ref_date is None:
            ref_date = state["date"]

        LOG.info(f"Reference date set to {ref_date}")

        # TODO: not ideal, but state["date"] is not timezone aware
        return ref_date.replace(tzinfo=None)

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def _create_field_var(self, name: str) -> Variable:
        """Create a variable with missing values"""
        assert self.ncfile is not None

        if metadata := self.variable_metadata.get(name):
            var_name = metadata.name
            attrs = metadata.attrs
        else:
            var_name = name
            attrs = {}

        var = self.ncfile.createVariable(
            var_name,
            self.float_size,
            self.dimensions,
            fill_value=self.missing_value,
            **self.compression,
        )

        # Set metadata based attributes
        var.setncatts(attrs)

        # Set output based attributes
        var.missing_value = self.missing_value
        var.fill_value = self.missing_value
        if self.proj_str is not None:
            var.grid_mapping = "projection"
            var.coordinates = "latitude longitude"

        LOG.info(f"Created variable {var_name}")
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

        self.reference_date = self._get_reference_date(state)

        state = self.post_process(state)

        # keep time unlimited
        time = None

        # Create new NetCDF file at self.path
        with LOCK:
            if self.ncfile is not None:
                return

            # TODO: add height dimensions and variables?
            self.ncfile = Dataset(self.path, "w", format="NETCDF4")

            # TODO: i4 or f8 for time?
            self.ncfile.createDimension("time", time)
            self.time = self.ncfile.createVariable("time", "f8", ("time",))
            self.time.units = f"seconds since {self.reference_date}"
            self.time.standard_name = "time"

            var = self.ncfile.createVariable("forecast_reference_time", "f8")
            var.units = f"seconds since {self.reference_date}"
            var.standard_name = "forecast_reference_time"

            # TODO: make sure this is right?
            var[:] = (state["date"] - self.reference_date).total_seconds()

        # TODO: this should already be available in the checkpoint
        output_template: Optional[str] = getattr(
            self.context.development_hacks, "output_template", None
        )

        input_field_shape: Optional[tuple[int, int]] = getattr(
            self.context.development_hacks, "field_shape", None
        )

        if output_template is not None:
            LOG.info(output_template)
            with xr.open_zarr(output_template, consolidated=False) as template:
                if input_field_shape is not None:
                    assert len(input_field_shape) == 2, "Expected 2D field shape (y, x)"
                    self.field_shape = input_field_shape
                    (y_size, x_size) = self.field_shape
                else:
                    self.field_shape = template.field_shape
                    (y_size, x_size) = self.field_shape

                lats = np.reshape(template.latitudes.values, self.field_shape)
                lons = np.reshape(template.longitudes.values, self.field_shape)

                with LOCK:
                    self.ncfile.createDimension("y", y_size)
                    self.ncfile.createDimension("x", x_size)
                    self.ncfile.createDimension("ensemble_member", self.members)
                    var = self.ncfile.createVariable(
                        "ensemble_member", "i4", ("ensemble_member",)
                    )
                    var[:] = np.arange(self.members, dtype=np.int32)

                if self.proj_str is not None:
                    self._create_projections(lats, lons)

                self.dimensions = ("time", "ensemble_member", "y", "x")
                coord_dims = ("y", "x")

                log_str = f"y={y_size}, x={x_size}"
        else:
            lats = state["latitudes"]
            lons = state["longitudes"]
            values = len(lats)

            coord_dims = ("values",)

            self.field_shape = (values,)
            self.dimensions = ("time", "values")
            self.ncfile.createDimension("values", values)
            log_str = f"{values=}"

        with LOCK:
            var = self.ncfile.createVariable("latitude", "f8", coord_dims)
            var.standard_name = "latitude"
            var.units = "degrees_north"
            var[:] = lats

            var = self.ncfile.createVariable("longitude", "f8", coord_dims)
            var.standard_name = "longitude"
            var.units = "degrees_east"
            var[:] = lons

        LOG.info(
            f"Created NetCDF file {self.path} with dimensions: "
            f"(time=unlimited, {log_str})"
        )

    def _create_projections(self, lats, lons):
        assert self.ncfile is not None
        assert self.proj_str is not None

        # Convert projection string to dictionary of attributes
        crs = pyproj.CRS.from_proj4(self.proj_str)

        attrs = {
            k: v for k, v in crs.to_cf().items() if v != "unknown" and k != "crs_wkt"
        }
        attrs["earth_radius"] = 6_371_000.0

        x, y = self._get_projections(lats, lons)
        with LOCK:
            var = self.ncfile.createVariable("x", "f4", ("x",))
            var.standard_name = "projection_x_coordinate"
            var.units = "m"
            var[:] = x

            var = self.ncfile.createVariable("y", "f4", ("y",))
            var.standard_name = "projection_y_coordinate"
            var.units = "m"
            var[:] = y

            # Create and set attributes to the projection var
            var = self.ncfile.createVariable("projection", "i4", [])
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

    def _get_projections(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reverse engineer x and y vectors from lats and lons."""
        assert self.proj_str is not None

        lonlat_proj = "proj+=longlat"
        proj_from = pyproj.Proj(lonlat_proj)
        proj_to = pyproj.Proj(self.proj_str)

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

        step = state["date"] - self.reference_date
        self.time[self.n] = step.total_seconds()

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            with LOCK:
                # value has shape (n_members, values), we need to reshape to (n_members, y_dim, x_dim)
                field = np.reshape(value, (-1, *self.field_shape))
                LOG.debug(f"🚧🚧🚧🚧🚧🚧 Writing {name}, {self.n}, {self.field_shape}")
                self.vars[name][self.n, 0, ...] = field

        # TODO: should this be synced here to avoid possible loss of data if job is interrupted?
        # self.ncfile.sync()

        self.n += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
