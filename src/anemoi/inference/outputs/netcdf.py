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
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import ensure_path
from ..decorators import main_argument
from ..output import Output
from ..utils.templating import render_template
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
        split_output: bool = False,
        strftime: str = "%Y%m%d%H%M",
        ekd_compatible: bool = False,
    ) -> None:
        """Initialise the NetCDF output object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        path : Path
            The path to save the NetCDF file to.
            If the parent directory does not exist, it will be created.
            When ``split_output`` is enabled, this may be a template such as
            ``"netcdf/{dateTime}_{step:03}.nc"``.
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
        split_output : bool, optional
            Whether to create one NetCDF file per written step.
        strftime : str, optional
            Datetime format used for the ``dateTime`` template variable.
        ekd_compatible : bool, optional
            When true, write an xarray-style NetCDF layout with ``date`` and
            ``latitude``/``longitude`` coordinates intended to be closer to the
            NetCDF shape accepted by Earthkit's NetCDF input path. Currently
            this requires ``split_output``.
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
        self.split_output = split_output
        self.strftime = strftime
        self.ekd_compatible = ekd_compatible
        self.n = 0
        self.vars: dict[str, object] = {}

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def _set_reference_date(self, state: State) -> None:
        """Set the reference date used for the NetCDF time coordinate."""
        reference_date = getattr(self.context, "reference_date", None)
        if reference_date is not None:
            self.reference_date = reference_date
            return

        step = state.get("step", datetime.timedelta(0))
        if isinstance(step, datetime.timedelta):
            self.reference_date = state["date"] - step
        else:
            self.reference_date = state["date"]

    def _step_value(self, step: object) -> int | float | object:
        """Convert timedelta steps to hours for path templating."""
        if not isinstance(step, datetime.timedelta):
            return step

        hours = step.total_seconds() / 3600
        if hours.is_integer():
            return int(hours)
        return hours

    def _template_context(self, state: State) -> dict[str, object]:
        """Return filename template values for a state."""
        date = state["date"]
        step = state.get("step", datetime.timedelta(0))
        if isinstance(step, datetime.timedelta):
            basetime = date - step
        else:
            basetime = self.reference_date or date

        return {
            "date": int(date.strftime("%Y%m%d")),
            "time": date.hour * 100 + date.minute,
            "dateTime": date.strftime(self.strftime),
            "step": self._step_value(step),
            "valid_datetime": date,
            "basetime": basetime,
            "baseDateTime": basetime.strftime(self.strftime),
        }

    def _resolve_path(self, state: State) -> Path:
        """Resolve the output path for the given state."""
        if not self.split_output:
            return self.path

        path = Path(render_template(str(self.path), self._template_context(state)))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _xarray_dataset(self, state: State) -> object:
        """Build an xarray Dataset closer to Earthkit's documented NetCDF layout."""
        import xarray as xr

        step = state.get("step", datetime.timedelta(0))
        if isinstance(step, datetime.timedelta):
            base_date = state["date"] - step
        else:
            base_date = self.reference_date or state["date"]

        coords = {
            "time": np.array([np.datetime64(state["date"])]),
            "latitude": ("values", np.asarray(state["latitudes"], dtype=self.float_size)),
            "longitude": ("values", np.asarray(state["longitudes"], dtype=self.float_size)),
        }
        data_vars = {
            "forecast_reference_time": np.array(np.datetime64(base_date)),
        }

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            array = np.asarray(value)
            if array.ndim == 1:
                array = array[np.newaxis, :]
            elif array.ndim == 2 and array.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"ekd_compatible NetCDF output expects a single-step field for {name}, got shape {array.shape}"
                )

            data_vars[name] = (("time", "values"), array.astype(self.float_size, copy=False))

        dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        dataset["time"].attrs.update({"standard_name": "time", "long_name": "time", "axis": "T"})
        dataset["forecast_reference_time"].attrs.update(
            {"standard_name": "forecast_reference_time", "long_name": "forecast_reference_time"}
        )
        dataset["latitude"].attrs.update({"units": "degrees_north", "long_name": "latitude"})
        dataset["longitude"].attrs.update({"units": "degrees_east", "long_name": "longitude"})
        return dataset

    def _xarray_encoding(self, dataset: object) -> dict[str, dict[str, object]]:
        """Return NetCDF encodings for the xarray-compatible output."""
        reference = dataset["forecast_reference_time"].values
        if isinstance(reference, np.ndarray):
            reference = reference.item()
        reference = np.datetime64(reference, "s")
        reference_units = np.datetime_as_string(reference, unit="s").replace("T", " ")

        encoding = {
            "time": {"dtype": "i4", "units": f"hours since {reference_units}", "calendar": "proleptic_gregorian"},
            "latitude": {"dtype": self.float_size},
            "longitude": {"dtype": self.float_size},
        }

        for name in dataset.data_vars:
            if np.issubdtype(dataset[name].dtype, np.datetime64):
                encoding[name] = {}
                continue

            encoding[name] = {"dtype": self.float_size}
            if self.missing_value is not None:
                encoding[name]["_FillValue"] = self.missing_value

        return encoding

    def _write_xarray_step(self, state: State, path: Path) -> None:
        """Write one step using xarray's coordinate model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            os.remove(path)

        dataset = self._xarray_dataset(state)
        try:
            dataset.to_netcdf(path, encoding=self._xarray_encoding(dataset))
        finally:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()

    def _open_dataset(self, state: State, path: Path, time_size: int | None) -> None:
        """Open a NetCDF file and initialize dimensions and coordinates."""
        from netCDF4 import Dataset

        if self.ncfile is not None:
            self.close()

        if os.path.exists(path):
            os.remove(path)

        with LOCK:
            self.ncfile = Dataset(path, "w", format="NETCDF4")

        compression = {}
        values = len(state["latitudes"])

        with LOCK:
            self.values_dim = self.ncfile.createDimension("values", values)
            self.time_dim = self.ncfile.createDimension("time", time_size)
            self.time_var = self.ncfile.createVariable("time", "i4", ("time",), **compression)

            self.time_var.units = f"seconds since {self.reference_date}"
            self.time_var.long_name = "time"
            self.time_var.calendar = "gregorian"

            latitudes = state["latitudes"]
            self.latitude_var = self.ncfile.createVariable("latitude", self.float_size, ("values",), **compression)
            self.latitude_var.units = "degrees_north"
            self.latitude_var.long_name = "latitude"

            longitudes = state["longitudes"]
            self.longitude_var = self.ncfile.createVariable("longitude", self.float_size, ("values",), **compression)
            self.longitude_var.units = "degrees_east"
            self.longitude_var.long_name = "longitude"

            self.latitude_var[:] = latitudes
            self.longitude_var[:] = longitudes

        self.n = 0
        self.vars = {}

    def open(self, state: State) -> None:
        """Open the NetCDF file and initialize dimensions and variables.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        self._set_reference_date(state)

        if self.ekd_compatible:
            if not self.split_output:
                raise NotImplementedError("NetCDFOutput(ekd_compatible=True) currently requires split_output=True")
            return

        if self.split_output:
            return

        with LOCK:
            if self.ncfile is not None:
                return

        state = self.post_process(state)
        self._open_dataset(state, self.path, time_size=None)

    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        values = len(state["latitudes"])
        compression = {}

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

    def _write_current_step(self, state: State) -> None:
        """Write the already-open step to the current NetCDF file."""
        self.ensure_variables(state)

        step = state["date"] - self.reference_date
        self.time_var[self.n] = int(step.total_seconds())

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            with LOCK:
                LOG.debug("Writing %s at index %s with shape %s", name, self.n, value.shape)
                self.vars[name][self.n] = value

        self.n += 1

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        if self.reference_date is None:
            self._set_reference_date(state)

        if self.ekd_compatible:
            path = self._resolve_path(state)
            self._write_xarray_step(state, path)
            return

        if self.split_output:
            path = self._resolve_path(state)
            self._open_dataset(state, path, time_size=1)
            try:
                self._write_current_step(state)
            finally:
                self.close()
            return

        self._write_current_step(state)

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
