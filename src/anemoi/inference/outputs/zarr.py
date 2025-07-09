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
import shutil
from typing import Any
from typing import List
from typing import Literal
from typing import Optional

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def create_zarr_array(
    store,
    name: str,
    shape: tuple,
    dtype: str,
    dimensions: tuple[str, ...],
    chunks: tuple[int, ...] | Literal["auto"],
    fill_value: Optional[float] = None,
) -> Any:
    """Create a Zarr array with the given parameters.

    Parses the Zarr version to handle differences in API between versions 2 and 3.
    """
    import zarr

    zarr_version = int(zarr.__version__.split(".")[0])

    array = zarr.create_array(
        store,
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
        dimension_names=dimensions if zarr_version >= 3 else None,
        overwrite=True,
    )
    if zarr_version < 3:
        array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    return array


@output_registry.register("zarr")  # type: ignore
@main_argument("store")
class ZarrOutput(Output):
    """Zarr output class."""

    def __init__(
        self,
        context: Context,
        store: Any,
        variables: Optional[List[str]] = None,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
        missing_value: Optional[float] = np.nan,
        float_size: str = "f4",
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
    ) -> None:
        """Initialize the ZarrOutput object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        store : Any
            The Zarr store to save the data.
            Can be a file path or a Zarr store.
            If an existing store is provided, it is assumed to
            be a writable store and empty.
        variables : list, optional
            The list of variables to write, by default None.
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        float_size : str, optional
            The size of the float, by default "f4".
        missing_value : float, optional
            The missing value, by default np.nan.
        chunks : tuple[int, ...] | Literal['auto'], optional
            The chunk size for the Zarr arrays, by default 'auto'.
        """

        super().__init__(
            context, variables=variables, output_frequency=output_frequency, write_initial_state=write_initial_state
        )

        from zarr.storage import StoreLike

        self.zarr_store: StoreLike = store
        self.missing_value = missing_value
        self.chunks = chunks
        self.float_size = float_size

        self.extra_time = int(self.write_step_zero)
        self._vars = {}

    def __repr__(self) -> str:
        """Return a string representation of the ZarrOutput object."""
        return f"ZarrOutput({self.zarr_store})"

    def open(self, state: State) -> None:
        """Open the Zarr file and initialize dimensions.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        import zarr
        from zarr.storage import LocalStore

        if isinstance(self.zarr_store, str):
            if os.path.exists(self.zarr_store):
                LOG.warning(f"Zarr store {self.zarr_store} already exists. It will be overwritten.")
                shutil.rmtree(self.zarr_store)
            self.zarr_store = LocalStore(self.zarr_store)

        self.zarr_group = zarr.open_group(self.zarr_store, mode="w")

        values = len(state["latitudes"])

        time = 0
        self.reference_date = state["date"]
        if (time_step := getattr(self.context, "time_step", None)) and (
            lead_time := getattr(self.context, "lead_time", None)
        ):
            time = lead_time // time_step
            time += self.extra_time

        if reference_date := getattr(self.context, "reference_date", None):
            self.reference_date = reference_date

        self.time_size = time
        self.time_array = create_zarr_array(
            self.zarr_store,
            name="time",
            shape=(self.time_size,),
            dtype="i4",
            dimensions=("time",),
            chunks=self.chunks,
        )
        self.time_array.attrs.update(
            {
                "units": f"seconds since {self.reference_date}",
                "calendar": "gregorian",
            }
        )

        latitudes = state["latitudes"]
        self.latitude_var = create_zarr_array(
            self.zarr_store,
            name="latitude",
            shape=(values,),
            dtype=self.float_size,
            dimensions=("latitude",),
            chunks=self.chunks,
            fill_value=self.missing_value,
        )
        self.latitude_var.attrs.update({"units": "degrees_north", "long_name": "latitude"})

        longitudes = state["longitudes"]
        self.longitude_var = create_zarr_array(
            self.zarr_store,
            name="longitude",
            shape=(values,),
            dtype=self.float_size,
            dimensions=("longitude",),
            chunks=self.chunks,
            fill_value=self.missing_value,
        )

        self.n = 0
        self.latitude_var[:] = latitudes
        self.longitude_var[:] = longitudes

        self.n = 0

    def _variable_array(self, name: str, values_size: int) -> Any:
        """Get the variable array by name.

        Creates the variable array if it does not exist.

        Parameters
        ----------
        name : str
            The name of the variable.
        values_size : int
            The size of the values dimension.

        Returns
        -------
        Any
            The variable array.
        """
        if name in self._vars:
            return self._vars[name]

        self._vars[name] = create_zarr_array(
            self.zarr_store,
            name=name,
            shape=(self.time_size, values_size),
            dtype=self.float_size,
            dimensions=("time", "values"),
            chunks=self.chunks,
            fill_value=self.missing_value,
        )

        variable = self.typed_variables[name]
        self._vars[name].attrs.update({"param": variable.grib_keys.get("param", name)})
        return self._vars[name]

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        step = state["date"] - self.reference_date
        self.time_array[self.n] = step.total_seconds()

        values = len(state["latitudes"])

        for name, value in state["fields"].items():

            if self.skip_variable(name):
                continue

            self._variable_array(name, values)[self.n] = value
        self.n += 1

    def close(self) -> None:
        """Close the Zarr file."""
        import zarr
        from zarr.abc.store import Store

        if self.zarr_store is not None and isinstance(self.zarr_store, Store):
            zarr.consolidate_metadata(self.zarr_store)
            self.zarr_store.close()
            self.zarr_store = None
