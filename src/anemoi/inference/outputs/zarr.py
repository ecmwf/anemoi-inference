# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
import shutil
from typing import Any
from typing import Literal

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def create_zarr_array(
    store: Any,
    name: str,
    shape: tuple,
    dtype: str,
    dimensions: tuple[str, ...],
    chunks: tuple[int, ...] | Literal["auto"] | bool,
    fill_value: float | None = None,
) -> Any:
    """Create a Zarr array with the given parameters.

    Parses the Zarr version to handle differences in API between versions 2 and 3.
    """
    import zarr

    chunks = chunks if zarr.__version__ >= "3" else chunks if not chunks == "auto" else True

    store: zarr.Group = store

    if zarr.__version__ >= "3":
        from zarr import create_array
    else:

        def create_array(*, store, **kwargs):
            """Create a Zarr array using the zarr 2 API."""
            return store.create_dataset(**kwargs)

    array = create_array(
        store=store,
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
        dimension_names=dimensions,
        overwrite=True,
    )
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
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        missing_value: float | None = np.nan,
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

        if isinstance(self.zarr_store, str):
            if os.path.exists(self.zarr_store):
                LOG.warning(f"Zarr store {self.zarr_store} already exists. It will be overwritten.")
                shutil.rmtree(self.zarr_store)

            if zarr.__version__ >= "3":
                from zarr.storage import LocalStore

                self.zarr_store = LocalStore(self.zarr_store)
            else:
                from zarr.storage import DirectoryStore

                self.zarr_store = DirectoryStore(self.zarr_store)

        if zarr.__version__ >= "3":
            self.zarr_group = self.zarr_store
        else:
            self.zarr_group = zarr.open_group(self.zarr_store, mode="w")

        values = len(state["latitudes"])

        time = 0

        self.reference_date = state["date"]

        lead_time = getattr(self.context, "lead_time", None)
        time_step = self.context.checkpoint.timestep

        if lead_time is None:
            raise RuntimeError(
                "When setting up the ZarrOutput, the `lead_time` was not yet set on the context, therefore unable to construct the arrays."
            )

        time = lead_time // time_step
        time += int(self.write_step_zero)

        if reference_date := getattr(self.context, "reference_date", None):
            self.reference_date = reference_date

        if not self.write_step_zero and time_step is not None:
            self.reference_date -= time_step

        self.time_size = time
        self.time_array = create_zarr_array(
            self.zarr_group,
            name="time",
            shape=(self.time_size,),
            dtype="i4",
            dimensions=("time",),
            chunks=(1,),
        )
        self.time_array.attrs.update(
            {
                "units": f"seconds since {self.reference_date}",
                "calendar": "gregorian",
            }
        )

        latitudes = state["latitudes"]
        self.latitude_var = create_zarr_array(
            self.zarr_group,
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
            self.zarr_group,
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
            self.zarr_group,
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

    def write_initial_state(self, state: State) -> None:
        """Write the initial state to the Zarr file."""
        state = self.reduce(state)  # Reduce to only the last step

        return super().write_initial_state(state)

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

        if zarr.__version__ >= "3":
            from zarr.abc.store import Store
        else:
            from zarr.storage import BaseStore as Store

        if self.zarr_store is not None and not isinstance(self.zarr_store, str):
            if isinstance(self.zarr_store, Store):
                zarr.consolidate_metadata(self.zarr_store)
                self.zarr_store.close()
            self.zarr_store = None
