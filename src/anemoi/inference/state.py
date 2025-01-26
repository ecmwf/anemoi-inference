# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar

import numpy as np

if TYPE_CHECKING:
    import earthkit.data as ekd
    import xarray as xr

Array = TypeVar("Array")


def extract_keys(s: str) -> list[str]:
    # Use regular expression to find all occurrences of {key}
    keys = re.findall(r"\{(.*?)\}", s)
    return keys


def permute_dict(d: dict) -> list[dict]:
    # Get the keys and values from the dictionary
    keys = d.keys()
    values = d.values()

    # Generate all permutations using itertools.product
    permutations = itertools.product(*values)

    # Convert permutations into a list of dictionaries
    result = [dict(zip(keys, permutation)) for permutation in permutations]

    return result


def summarise_list(lst: list, max_length: int) -> str:
    if len(lst) > max_length:
        summary = f"{str(lst[:3])[:-1]} ... {str(lst[-3:])[1:]} (Total: {len(lst)} items)"
        return summary
    return str(lst)


class State(dict[str, np.ndarray]):
    """A collection of data for inference."""

    def __init__(self, data: dict[str, np.ndarray] = None, *, private_info: Any = None, **kwargs):
        """Create a State object.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dictionary of data to store in the State
        private_info : Any, optional
            Private info to pass with the State, by default None
        """
        super().__init__(data or {}, **kwargs)
        self.__private_info = private_info

    def __repr__(self):
        return f"State({summarise_list(list(self.keys()), 8)}, private_info = {self.__private_info})"

    def take(self, axis: int, indices: int | slice | tuple[int, ...], *, copy: bool = True) -> State:
        """Take data from the State with a specific dimension and index.

        Parameters
        ----------
        axis : int
            Dimension to take data from
        indices : int | slice | tuple[int, ...]
            Index to take data with
        copy : bool, optional
            Whether to copy the data, If False, the data will be taken without copying,
            by default True

        Returns
        -------
        State
            State object with the dimension and index taken on all keys

        Examples
        --------
        >>> data = np.random.rand(3, 2, 4)
        >>> names = ["a", "b", "c"]
        >>> state = State.from_numpy(data, names)
        >>> state.take(0, 0)
        State(['a', 'b', 'c'], private_info = None)
        >>> state.take(0, 0).shape
        {'a': (2, 4), 'b': (2, 4), 'c': (2, 4)}
        """
        if copy:
            return State(
                {key: value.take(indices, axis) for key, value in self.items()}, private_info=self.__private_info
            )

        index = [[slice(None)] * axis, indices, [slice(None)] * (len(self[list(self.keys())[0]].shape) - axis - 1)]
        index = [i for i in index if (isinstance(i, list) and len(i) > 1) or not isinstance(i, list)]
        return State({key: value.__getitem__(*index) for key, value in self.items()}, private_info=self.__private_info)

    def to_array(
        self,
        order: list[str],
        *,
        stack_function: Callable[[list[Array], Any], Array] = np.stack,
        array_function: Callable[[np.array], Array] = np.array,
        **kwargs,
    ) -> Array:
        """Convert the State to an array.

        Parameters
        ----------
        order : list[str]
            Order to extract the keys from the State
        stack_function : Callable, optional
            Function to stack arrays with, by default np.stack
        array_function: Callable, optional
            Function to convert arrays with, must take np.array, by default np.array
        **kwargs:
            Additional keyword arguments to pass to the stack_function

        Returns
        -------
        Array
            Stacked array

        Raises
        ------
        ValueError
            If any keys in order are not in the State

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(3, 2, 4)
        >>> names = ["a", "b", "c"]
        >>> state = State.from_numpy(data, names)
        >>> state.to_array(names[::-1]).shape
        (3, 2, 4)
        """
        if any(key not in self for key in order):
            raise ValueError("Some keys in order are not in the State", self.keys(), order)
        return stack_function([array_function(self.get(key)) for key in order], **kwargs)

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        """Get the shape of the State"""
        return {key: value.shape for key, value in self.items()}

    @classmethod
    def from_xarray(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        flatten: Optional[str] = None,
        variable_dim: str = "variable",
        private_info: Any = None,
    ) -> State:
        """Convert an xarray Dataset or DataArray to a State.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            xr.Dataset or xr.DataArray object to convert to a State
        flatten : str | None, optional
            F-string dictating how to flatten dimensions, by default None
            E.g. "{variable}_{level}" will create a new key for each level value
        variable_dim : str, optional
            Dimension name for variables if xr.DataArray given, by default "variable"
        private_info : Any, optional
            Private info to pass to State, by default None

        Returns
        -------
        State
            State object

        Raises
        ------
        ValueError
            If no keys are found in flatten
        KeyError
            If variable_dim is not found in keys

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> ds
        <xarray.Dataset> Size: 31MB
        Dimensions:  (lat: 25, time: 2920, lon: 53)
        Coordinates:
        * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
        * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
        * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
        Data variables:
            air      (time, lat, lon) float64 31MB ...
        Attributes:
            Conventions:  COARDS
            title:        4x daily NMC reanalysis (1948)
            description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
            platform:     Model
            references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
        >>> State.from_xarray(ds))
        State(['air'], private_info = None)
        >>> State.from_xarray(ds, flatten="{variable}_{lat}"))
        State(['air_75.0', 'air_72.5', 'air_70.0', 'air_67.5', 'air_65.0', 'air_62.5', 'air_60.0', 'air_57.5', 'air_55.0', 'air_52.5', 'air_50.0', 'air_47.5', 'air_45.0', 'air_42.5', 'air_40.0', 'air_37.5', 'air_35.0', 'air_32.5', 'air_30.0', 'air_27.5', 'air_25.0', 'air_22.5', 'air_20.0', 'air_17.5', 'air_15.0'], private_info = None)
        """
        import xarray as xr

        # Flatten the data if required
        if flatten is not None:
            keys = extract_keys(flatten)
            if len(keys) == 0:
                raise ValueError("No keys found in flatten")

            if variable_dim not in keys:
                raise KeyError(f"variable_dim {variable_dim} not found in keys {keys}")

            new_data = xr.Dataset()
            for var in data.coords[variable_dim].values:
                var_ds = data.sel({variable_dim: var})

                if not any(key in var_ds.dims for key in keys):
                    new_data[var] = data.sel({variable_dim: var})

                for perm in permute_dict({key: list(np.atleast_1d(var_ds.coords[key].values)) for key in keys}):
                    new_data[flatten.format(**perm)] = data.sel({variable_dim: var, **perm})

            data = new_data

        # Convert to DataArray if Dataset so dimensions are in the correct order
        if isinstance(data, xr.Dataset):
            data = data.to_dataarray(dim=variable_dim)

        dims = list(data.dims)
        dims.remove(variable_dim)
        data = data.transpose(variable_dim, *dims)

        # Get all variables and their values in the data
        variable_dict = {}
        for var in data.coords[variable_dim].values:
            variable_dict[var] = data.sel({variable_dim: var}).values
        return State(variable_dict, private_info=private_info)

    @classmethod
    def from_earthkit(self, fieldlist: "ekd.FieldList", private_info: Any = None, **kwargs) -> State:
        """Convert a FieldList to a State.

        Parameters
        ----------
        fieldlist : ekd.FieldList
            earthkit data FieldList object
        private_info : Any
            Private information to store in the State object
        **kwargs:
            Additional keyword arguments to pass to the to_xarray method of the FieldList object.
            See /earthkit/data/utils/xarray/engine.py/EarthkitBackendEntrypoint/open_dataset for more information

        Returns
        -------
        State
            State object

        Examples
        --------
        >>> import earthkit.data as ekd
        >>> ekd.download_example_file("test6.grib")
        >>> fieldlist = ekd.from_source("file", "test6.grib")
        >>> fieldlist
        GRIBReader(test6.grib)
        >>> State.from_earthkit(fieldlist)
        State(['t', 'u', 'v'], private_info = None)
        >>> State.from_earthkit(fieldlist, variable_key="par_lev_type", remapping={"par_lev_type": "{param}_{levelist}"})
        State(['t_1000', 't_850', 'u_1000', 'u_850', 'v_1000', 'v_850'], private_info = None)
        """
        return self.from_xarray(fieldlist.to_xarray(**kwargs), private_info=private_info)

    @classmethod
    def from_numpy(
        self,
        data: np.ndarray,
        names: list[str],
        *,
        axis: int = 0,
        private_info: Any = None,
    ) -> State:
        """Convert a numpy array to a State.

        Parameters
        ----------
        data : np.ndarray
            Numpy array to convert to a State
        names : list[str]
            Names upon `axis` to use as keys
        axis : int, optional
            Axis to split data upon, by default 0
        private_info : Any, optional
            Private information to store in the State object, by default None

        Returns
        -------
        State
            State object

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(3, 2, 4)
        >>> names = ["a", "b", "c"]
        >>> State.from_numpy(data, names)
        State(['a', 'b', 'c'], private_info = None)
        >>> names = ["a", "b", "c", "d"]
        >> State.from_numpy(data, names, axis=2)
        State(['a', 'b', 'c', 'd'], private_info = None)
        """

        if axis != 0:
            data = np.moveaxis(data, axis, 0)
        return State(dict(zip(names, data)), private_info=private_info)

    def copy(self):
        """Copy the State object."""
        return State(dict(self.items()), private_info=self.__private_info)
