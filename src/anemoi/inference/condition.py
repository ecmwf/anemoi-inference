from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
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


class Condition(dict):
    """A collection of data for inference."""

    def __init__(self, data: dict[str, np.ndarray] = None, *, private_info: Any = None, **kwargs):
        """Create a Condition object.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dictionary of data to store in the Condition
        private_info : Any, optional
            Private info to pass with the Condition, by default None
        """
        super().__init__(data or {}, **kwargs)
        self.__private_info = private_info

    def __repr__(self):
        return f"Condition({summarise_list(list(self.keys()), 8)}, private_info = {self.__private_info})"

    def to_array(
        self,
        order: list[str],
        *,
        stack_function: Callable[[list[Array], Any], Array] = np.stack,
        array_function: Callable[[np.array], Array] = np.array,
        **kwargs,
    ) -> Array:
        """Convert the Condition to an array.

        Parameters
        ----------
        order : list[str]
            Order to extract the keys from the Condition
        stack_function : Callable, optional
            Function to stack arrays with, by default np.stack
        array_function: Callable, optional
            Function to convert arrays with, must take np.array, by default np.array
        **kwargs:
            Additional keyword arguments to pass to the stack_function

        Returns
        -------
        T
            Stacked array

        Raises
        ------
        ValueError
            If any keys in order are not in the Condition

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(3, 2, 4)
        >>> names = ["a", "b", "c"]
        >>> condition = Condition.from_numpy(data, names)
        >>> condition.to_array(names[::-1]).shape
        (3, 2, 4)
        """
        if any(key not in self for key in order):
            raise ValueError("Some keys in order are not in the Condition", self.keys(), order)
        return stack_function([array_function(self.get(key)) for key in order], **kwargs)

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        """Get the shape of the Condition"""
        return {key: value.shape for key, value in self.items()}

    @classmethod
    def from_xarray(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        flatten: str | None = None,
        variable_dim: str = "variable",
        private_info: Any = None,
    ) -> Condition:
        """Convert an xarray Dataset or DataArray to a Condition.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            xr.Dataset or xr.DataArray object to convert to a Condition
        flatten : str | None, optional
            F-string dictating how to flatten dimensions, by default None
            E.g. "{variable}_{level}" will create a new key for each level value
        variable_dim : str, optional
            Dimension name for variables if xr.DataArray given, by default "variable"
        private_info : Any, optional
            Private info to pass to Condition, by default None

        Returns
        -------
        Condition
            Condition object

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
        >>> Condition.from_xarray(ds))
        Condition(['air'], private_info = None)
        >>> Condition.from_xarray(ds, flatten="{variable}_{lat}"))
        Condition(['air_75.0', 'air_72.5', 'air_70.0', 'air_67.5', 'air_65.0', 'air_62.5', 'air_60.0', 'air_57.5', 'air_55.0', 'air_52.5', 'air_50.0', 'air_47.5', 'air_45.0', 'air_42.5', 'air_40.0', 'air_37.5', 'air_35.0', 'air_32.5', 'air_30.0', 'air_27.5', 'air_25.0', 'air_22.5', 'air_20.0', 'air_17.5', 'air_15.0'], private_info = None)
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
        return Condition(variable_dict, private_info=private_info)

    @classmethod
    def from_earthkit(self, fieldlist: "ekd.FieldList", private_info: Any = None, **kwargs) -> Condition:
        """Convert a FieldList to a Condition.

        Parameters
        ----------
        fieldlist : ekd.FieldList
            earthkit data FieldList object
        private_info : Any
            Private information to store in the Condition object
        **kwargs:
            Additional keyword arguments to pass to the to_xarray method of the FieldList object.
            See /earthkit/data/utils/xarray/engine.py/EarthkitBackendEntrypoint/open_dataset for more information

        Returns
        -------
        Condition
            Condition object

        Examples
        --------
        >>> import earthkit.data as ekd
        >>> ekd.download_example_file("test6.grib")
        >>> fieldlist = ekd.from_source("file", "test6.grib")
        >>> fieldlist
        GRIBReader(test6.grib)
        >>> Condition.from_earthkit(fieldlist)
        Condition(['t', 'u', 'v'], private_info = None)
        >>> Condition.from_earthkit(fieldlist, variable_key="par_lev_type", remapping={"par_lev_type": "{param}_{levelist}"})
        Condition(['t_1000', 't_850', 'u_1000', 'u_850', 'v_1000', 'v_850'], private_info = None)
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
    ) -> Condition:
        """Convert a numpy array to a Condition.

        Parameters
        ----------
        data : np.ndarray
            Numpy array to convert to a Condition
        names : list[str]
            Names upon `axis` to use as keys
        axis : int, optional
            Axis to split data upon, by default 0
        private_info : Any, optional
            Private information to store in the Condition object, by default None

        Returns
        -------
        Condition
            Condition object

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(3, 2, 4)
        >>> names = ["a", "b", "c"]
        >>> Condition.from_numpy(data, names)
        Condition(['a', 'b', 'c'], private_info = None)
        >>> names = ["a", "b", "c", "d"]
        >> Condition.from_numpy(data, names, axis=2)
        Condition(['a', 'b', 'c', 'd'], private_info = None)
        """

        if axis != 0:
            data = np.moveaxis(data, axis, 0)
        return Condition(dict(zip(names, data)), private_info=private_info)

    def copy(self):
        """Copy the Condition object."""
        return Condition(dict(self.items()), private_info=self.__private_info)
