# (C) Copyright 2024-2026 Anemoi contributors.
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
import warnings
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from functools import cached_property
from types import MappingProxyType as frozendict
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import earthkit.data as ekd
import numpy as np
from anemoi.metadata import Metadata as PkgMetadata
from anemoi.metadata import MetadataRegistry
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.provenance import gather_provenance_info
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray

if TYPE_CHECKING:
    from anemoi.transform.variables import Variable
    from earthkit.data import FieldList

LOG = logging.getLogger(__name__)


VARIABLE_CATEGORIES = {
    "computed",
    "forcing",
    "diagnostic",
    "prognostic",
    "constant",
    "accumulation",
}


def _remove_full_paths(x: Any) -> Any:
    """Remove full paths from the given data structure.

    Parameters
    ----------
    x : Any
        The data structure to process.

    Returns
    -------
    Any
        The processed data structure with full paths removed.
    """
    if isinstance(x, dict):
        return {k: _remove_full_paths(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_remove_full_paths(v) for v in x]
    if isinstance(x, str):
        return os.path.splitext(os.path.basename(x))[0]
    return x


class Metadata:
    """Metadata class backed by anemoi-metadata.

    ``self._metadata`` (a :class:`anemoi.metadata.Metadata` instance) is the sole
    source of truth.  There are no DotDict-based fallback paths.

    Supports both single-dataset and multi-dataset checkpoints.  For
    multi-dataset checkpoints, ``dataset_name`` selects which dataset's
    variables and indices are exposed.  Supporting arrays may be supplied in
    either flat format (``{name: array}``) or nested format
    (``{dataset_name: {name: array}}``); the constructor normalises them to
    the flat format expected internally.
    """

    def __init__(
        self,
        metadata: PkgMetadata | dict,
        supporting_arrays: dict | None = None,
        dataset_name: str = "data",
    ):
        """Initialize the Metadata object.

        Parameters
        ----------
        metadata : anemoi.metadata.Metadata | dict
            The anemoi-metadata package instance.  Must not be ``None``.
        supporting_arrays : dict, optional
            Supporting arrays.  May be flat (``{name: array}``) for legacy
            checkpoints or nested by dataset name
            (``{dataset_name: {name: array}}``) for modern checkpoints.
            The constructor normalises these to the flat format used internally.
        dataset_name : str, optional
            The dataset name to use, by default ``"data"``.
        """
        if isinstance(metadata, dict):
            metadata = PkgMetadata(MetadataRegistry.load(metadata, allow_stop=True))
        elif not isinstance(metadata, PkgMetadata):
            # Assume it's a raw MetadataContract instance; wrap it.
            metadata = PkgMetadata(metadata)

        assert isinstance(
            metadata, PkgMetadata
        ), f"metadata must be an anemoi.metadata.Metadata instance, got {type(metadata)}"
        self._metadata = metadata
        self.dataset_name = dataset_name

        if supporting_arrays is None:
            supporting_arrays = {}

        # Normalise supporting arrays: if the dict is nested (has a key equal
        # to dataset_name whose value is itself a dict), extract the inner dict.
        assert isinstance(supporting_arrays, dict)
        inner = supporting_arrays.get(dataset_name)
        if isinstance(inner, dict):
            self._supporting_arrays = inner
        else:
            self._supporting_arrays = supporting_arrays

        self._variables_categories = None

    def __repr__(self) -> str:
        return f"Metadata(name='{self.dataset_name}')"

    @property
    def multi_dataset(self) -> bool:
        return self._metadata.original_schema_version != "0.0"

    ###########################################################################
    # Debugging
    ###########################################################################

    def print_indices(self, print=LOG.info) -> None:
        """Print variable-to-tensor-index mappings for debugging."""
        print("Input variable indices:")
        for name, idx in sorted(self.variable_to_input_tensor_index.items(), key=lambda x: x[1]):
            print(f"  [{idx:3d}] {name}")

        print("")
        print("Output variable indices:")
        for name, idx in sorted(self.variable_to_output_tensor_index.items(), key=lambda x: x[1]):
            print(f"  [{idx:3d}] {name}")

        print("")
        print("Variable categories:")
        self.print_variable_categories(print=print)

    ###########################################################################
    # Inference
    ###########################################################################

    @cached_property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields."""
        result = list(range(0, self.multi_step_input))
        result = [-s * self.timestep for s in result]
        return sorted(result)

    @cached_property
    def timestep(self) -> datetime.timedelta:
        """Model time stepping timestep."""
        return to_timedelta(self._metadata.timestep)

    @cached_property
    def precision(self) -> str | int:
        """Return the precision of the model (bits per float)."""
        return self._metadata.precision

    @cached_property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (
            1,
            self.multi_step_input,
            self.number_of_grid_points,
            len(self.input_tensor_index_to_variable),
        )

    @cached_property
    def output_shape(
        self,
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, int]:
        # Multi-step output checkpoints have an extra time dimension
        if self.multi_step_output > 1:
            return (
                1,
                self.multi_step_output,
                1,
                self.number_of_grid_points,
                len(self.output_tensor_index_to_variable),
            )
        return (
            1,
            1,
            self.number_of_grid_points,
            len(self.output_tensor_index_to_variable),
        )

    def _make_indices_mapping(self, indices_from: list, indices_to: list) -> frozendict:
        """Create a mapping between two sets of indices.

        Parameters
        ----------
        indices_from : list
            The source indices.
        indices_to : list
            The target indices.

        Returns
        -------
        frozendict
            The mapping between the source and target indices.
        """
        assert len(indices_from) == len(indices_to), (indices_from, indices_to)
        return frozendict({i: j for i, j in zip(indices_from, indices_to)})

    @cached_property
    def variable_to_input_tensor_index(self) -> frozendict:
        """Return the mapping between variable name and input tensor index."""
        return frozendict(self._metadata.raw.get_variable_indices(self.dataset_name))

    @cached_property
    def variable_to_output_tensor_index(self) -> frozendict:
        """Return the mapping between variable name and output tensor index."""
        return frozendict(self._metadata.raw.get_output_variable_indices(self.dataset_name))

    @cached_property
    def input_tensor_index_to_variable(self) -> frozendict:
        """Return the mapping between input tensor index and variable name."""
        return frozendict({v: k for k, v in self.variable_to_input_tensor_index.items()})

    @cached_property
    def output_tensor_index_to_variable(self) -> frozendict:
        """Return the mapping between output tensor index and variable name."""
        return frozendict({v: k for k, v in self.variable_to_output_tensor_index.items()})

    @cached_property
    def number_of_grid_points(self) -> int:
        """Return the number of grid points per fields."""
        # Supporting arrays take priority (e.g. LAM grid_indices defines the cutout)
        if "grid_indices" in self._supporting_arrays:
            return len(self.load_supporting_array("grid_indices"))
        if "latitudes" in self._supporting_arrays:
            return len(self.load_supporting_array("latitudes"))
        if self._metadata.grid_points is not None:
            return self._metadata.grid_points
        raise ValueError(
            "Cannot determine number of grid points: "
            "no supporting arrays ('grid_indices' or 'latitudes') and "
            "anemoi-metadata did not provide grid_points."
        )

    @cached_property
    def number_of_input_features(self) -> int:
        """Return the number of input features."""
        return len(self.variable_to_input_tensor_index)

    @cached_property
    def model_computed_variables(self) -> tuple:
        """The initial conditions variables that need to be computed and not retrieved."""
        return tuple(self._metadata.computed_forcings(self.dataset_name))

    @cached_property
    def multi_step_input(self) -> int:
        """Number of past steps needed for the initial conditions tensor."""
        return self._metadata.multi_step_input

    @cached_property
    def multi_step_output(self) -> int:
        """Number of future steps predicted by single model forward."""
        return self._metadata.multi_step_output

    @cached_property
    def prognostic_output_mask(self) -> IntArray:
        """Return the prognostic output mask."""
        output_vars = self._metadata.raw.get_variable_types(self.dataset_name).get("prognostic", [])
        output_index = self.variable_to_output_tensor_index
        return np.array(sorted(output_index[v] for v in output_vars if v in output_index))

    @cached_property
    def prognostic_input_mask(self) -> IntArray:
        """Return the prognostic input mask."""
        prognostic_vars = self._metadata.raw.get_variable_types(self.dataset_name).get("prognostic", [])
        input_index = self.variable_to_input_tensor_index
        return np.array(sorted(input_index[v] for v in prognostic_vars if v in input_index))

    def has_supporting_array(self, name: str) -> bool:
        """Check if the metadata has a supporting array with the given name.

        Parameters
        ----------
        name : str
            The name of the supporting array.

        Returns
        -------
        bool
            True if the supporting array exists, False otherwise.
        """
        return name in self._supporting_arrays

    ###########################################################################
    # Variables
    ###########################################################################

    @cached_property
    def dataset_names(self) -> list[str]:
        """Return the list of dataset names."""
        return self._metadata.dataset_names

    @cached_property
    def task(self) -> str | None:
        """Return the task label."""
        return self._metadata.task

    @property
    def variables(self) -> tuple:
        """Return the variables as found in the training dataset."""
        return tuple(self._metadata.raw.get_variable_indices(self.dataset_name).keys())

    @cached_property
    def variables_metadata(self) -> dict[str, Any]:
        """Return the variables and their metadata as found in the training dataset."""
        return self._metadata.variables_metadata(self.dataset_name)

    @cached_property
    def prognostic_variables(self) -> list:
        """Variables that are marked as prognostic."""
        return self._metadata.raw.get_variable_types(self.dataset_name).get("prognostic", [])

    @cached_property
    def index_to_variable(self) -> frozendict:
        """Return a mapping from index to variable name."""
        return frozendict({i: v for i, v in enumerate(self.variables)})

    @cached_property
    def typed_variables(self) -> "dict[str, Variable]":
        """Returns strongly typed variables."""
        return self._metadata.typed_variables(self.dataset_name)

    @cached_property
    def accumulations(self) -> list:
        """Return the indices of the variables that are accumulations."""
        return list(self._metadata.accumulations(self.dataset_name))

    def name_fields(self, fields: ekd.FieldList, namer: Callable[..., str] | None = None) -> "FieldList":
        """Name fields using the provided namer.

        Parameters
        ----------
        fields : FieldList
            The fields to name.
        namer : callable, optional
            The namer function, by default None.

        Returns
        -------
        FieldList
            The named fields.
        """
        from earthkit.data.indexing.fieldlist import FieldArray

        if namer is None:
            namer = self.default_namer()

        def _name(field: ekd.Field, _: str, original_metadata: dict[str, Any]) -> str:
            return namer(field, original_metadata)

        return FieldArray([f.clone(name=_name) for f in fields])

    def sort_by_name(
        self,
        fields: ekd.FieldList,
        *args: Any,
        namer: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> ekd.FieldList:
        """Sort fields by name.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to sort.
        args : Any
            Additional arguments.
        namer : callable, optional
            The namer function, by default None.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ekd.FieldList
            The sorted fields.
        """
        fields = self.name_fields(fields, namer=namer)
        return fields.order_by("name", *args, **kwargs)

    ###########################################################################
    # Default namer
    ###########################################################################

    def default_namer(self, *args: Any, **kwargs: Any) -> Callable[..., str]:
        """Return a callable that can be used to name earthkit-data fields.

        Parameters
        ----------
        args : Any
            Additional arguments.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Callable
            The namer function.
        """
        assert len(args) == 0, args
        assert len(kwargs) == 0, kwargs

        def namer(field: ekd.Field, metadata: dict[str, Any]) -> str:
            # TODO: Return the `namer` used when building the dataset
            warnings.warn("🚧  TEMPORARY CODE 🚧: Use the remapping in the metadata")
            param, levelist, levtype = (
                metadata.get("param"),
                metadata.get("levelist"),
                metadata.get("levtype"),
            )

            # Bug in eccodes that returns levelist for single level fields in GRIB2
            if levtype in ("sfc", "o2d"):
                levelist = None

            if levelist is None:
                return param

            return f"{param}_{levelist}"

        return namer

    ###########################################################################
    # Data retrieval
    ###########################################################################

    @property
    def grid(self) -> str | None:
        """Return the grid information."""
        return self._metadata.data_request(self.dataset_name).get("grid")

    @property
    def area(self) -> str | None:
        """Return the area information."""
        return self._metadata.data_request(self.dataset_name).get("area")

    @property
    def data_frequency(self) -> Any:
        """Get the data frequency."""
        return self._metadata.data_frequency(self.dataset_name)

    @property
    def target_explicit_times(self) -> Any:
        """Return the target explicit times from the training configuration."""
        return list(self._metadata.raw.get_output_relative_date_indices(self.dataset_name))

    @property
    def input_explicit_times(self) -> Any:
        """Return the input explicit times from the training configuration."""
        return list(self._metadata.raw.get_input_relative_date_indices(self.dataset_name))

    def select_variables(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        has_mars_requests: bool = False,
    ) -> list[str]:
        """Get variables from input.

        Parameters
        ----------
        include: List[str]
            Categories to include. Supports compound expressions with '+'.
        exclude: List[str]
            Categories to exclude. Supports compound expressions with '+'.
        has_mars_requests: bool
            If True, only include variables that have MARS requests.

        Returns
        -------
        List[str]
            The list of variables.
        """
        result = self._metadata.select_variables(
            include=include,
            exclude=exclude,
            dataset_name=self.dataset_name,
        )

        if has_mars_requests:
            variables_metadata = self.variables_metadata
            result = [v for v in result if "mars" in variables_metadata.get(v, {})]

        return result

    def variables_mask(self, *, variables: list[str]) -> IntArray:

        variable_to_input_tensor_index = self.variable_to_input_tensor_index
        indices = [variable_to_input_tensor_index[v] for v in variables]

        return np.array(indices)

    def select_variables_and_masks(
        self, *, include: list[str] | None = None, exclude: list[str] | None = None
    ) -> tuple[list[str], IntArray]:
        variables = self.select_variables(include=include, exclude=exclude, has_mars_requests=False)
        return variables, self.variables_mask(variables=variables)

    def mars_input_requests(self) -> Iterator[DataRequest]:
        """Generate MARS input requests.

        Returns
        -------
        Iterator[DataRequest]
            The MARS requests.
        """

        for variable in self.select_variables(include=["prognostic", "forcing"], exclude=["computed", "diagnostic"]):
            metadata = self.variables_metadata[variable]

            yield metadata["mars"].copy()

    def mars_by_levtype(self, levtype: str) -> tuple[set, set]:
        """Get MARS parameters and levels by levtype.

        Parameters
        ----------
        levtype : str
            The levtype to filter by.

        Returns
        -------
        tuple
            The parameters and levels.
        """

        params = set()
        levels = set()

        for variable in self.select_variables(
            include=["prognostic", "forcing"],
            exclude=["computed", "diagnostic"],
        ):
            metadata = self.variables_metadata[variable]

            mars = metadata["mars"]
            if mars.get("levtype") != levtype:
                continue

            if "param" in mars:
                params.add(mars["param"])

            if "levelist" in mars:
                levels.add(mars["levelist"])

        return params, levels

    def mars_requests(
        self,
        *,
        variables: list[str],
        dates: list[Date],
        use_grib_paramid: bool = False,
        always_split_time: bool = False,
        patch_request: Callable[[DataRequest], DataRequest] | None = None,
        dont_fail_for_missing_paramid: bool = False,
        **kwargs: Any,
    ) -> list[DataRequest]:
        """Generate MARS requests for the given variables and dates.

        Parameters
        ----------
        variables : list[str]
            The list of variables.
        dates : list[Date]
            The list of dates.
        use_grib_paramid : bool, optional
            Whether to use GRIB paramid, by default False.
        always_split_time : bool, optional
            Whether to always split time, by default False.
        patch_request : Optional[Callable], optional
            A callable to patch the request, by default None.
        dont_fail_for_missing_paramid : bool, optional
            Whether to not fail for missing param ids, by default False.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        List[DataRequest]
            The list of MARS requests.
        """
        # TODO: this code should could somewhere else
        from anemoi.utils.grib import shortname_to_paramid
        from earthkit.data.utils.availability import Availability

        assert variables, "No variables provided"

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        assert dates, "No dates provided"

        DEFAULT_KEYS = ("class", "expver", "type", "stream", "levtype")
        DEFAULT_KEYS_AND_TIME = ("class", "expver", "type", "stream", "levtype", "time")

        # ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC
        # The split oper/scda is a bit special

        # CHANGE: Avoid duplicate GRIB values when training on forecasts by removing the time dependency in KEYS.
        # Set always_split_time=True to restore it.
        # TODO: revisit this code, can probably be removed
        KEYS = {("oper", "fc"): DEFAULT_KEYS, ("scda", "fc"): DEFAULT_KEYS}

        requests: dict = defaultdict(list)
        for r in self.simple_mars_requests(variables=variables):
            for date in dates:
                r = r.copy()

                base = date

                r["date"] = base.strftime("%Y-%m-%d")
                r["time"] = base.strftime("%H%M")

                r.update(kwargs)  # We do it here so that the Availability can use that information

                if always_split_time:
                    keys = DEFAULT_KEYS_AND_TIME
                else:
                    keys = KEYS.get((r.get("stream"), r.get("type")), DEFAULT_KEYS)
                key = tuple(r.get(k) for k in keys)

                # Special case because of oper/scda

                requests[key].append(r)

        result: list[DataRequest] = []
        for reqs in requests.values():
            compressed = Availability(reqs)
            for r in compressed.iterate():
                if not r:
                    continue

                r = r.copy()

                # Convert all to lists
                for k, v in r.items():
                    if not isinstance(v, (list, tuple, set)):
                        v = [v]
                    r[k] = sorted(set(v))

                # Patch BEFORE the shortname to paramid
                if patch_request:
                    r = patch_request(r)

                # Convert all to lists (again)
                for k, v in r.items():
                    if not isinstance(v, (list, tuple, set)):
                        v = [v]
                    r[k] = sorted(set(v))

                if use_grib_paramid and "param" in r:

                    def shortname_to_paramid_no_fail(x: str) -> str:
                        try:
                            return shortname_to_paramid(x)
                        except KeyError:
                            LOG.warning("Could not convert shortname '%s' to paramid", x)
                            return x

                    if dont_fail_for_missing_paramid:
                        _ = shortname_to_paramid_no_fail
                    else:
                        _ = shortname_to_paramid

                    r["param"] = [_(p) for p in r["param"]]

                # Simplify the request

                for k in list(r.keys()):
                    v = r[k]
                    if len(v) == 1:
                        v = v[0]

                    # Remove empty values for when tree is not fully defined
                    if v == "-":
                        r.pop(k)
                        continue
                    r[k] = v
                result.append(r)

        return result

    def simple_mars_requests(self, *, variables: list[str]) -> Iterator[DataRequest]:
        """Generate MARS requests for the given variables.

        Parameters
        ----------
        variables : list
            The list of variables.

        Returns
        -------
        Iterator[DataRequest]
            The MARS requests.

        Raises
        ------
        ValueError
            If no variables are requested or if a variable is not found in the metadata.
        """
        if len(variables) == 0:
            raise ValueError("No variables requested")

        for variable in variables:

            if variable not in self.variables_metadata:
                raise ValueError(f"Variable {variable} not found in the metadata")

            if "mars" not in self.variables_metadata[variable]:
                raise ValueError(f"Variable {variable} has no MARS metadata")

            mars = self.variables_metadata[variable]["mars"].copy()

            for k in ("date", "hdate", "time", "valid_datetime", "variable"):
                mars.pop(k, None)

            yield mars

    ###########################################################################
    # Error reporting
    ###########################################################################

    def report_error(self) -> None:
        """Report an error with provenance information."""
        provenance = self.provenance_training()

        def _print(title: str, provenance: dict[str, Any]) -> None:
            LOG.info("")
            LOG.info("%s:", title)
            for package, git in sorted(provenance.get("git_versions", {}).items()):
                if package.startswith("anemoi."):
                    sha1 = git.get("git", {}).get("sha1", "unknown")
                    LOG.info(f"   {package:20}: {sha1}")

            for package, version in sorted(provenance.get("module_versions", {}).items()):
                if isinstance(version, dict) and "version" in version:
                    version = version["version"]
                if package.startswith("anemoi."):
                    LOG.info(f"   {package:20}: {version}")

        _print("Environment used during training", provenance)
        _print("This environment", gather_provenance_info())

        LOG.warning("If you are running from a git repository, the versions reported above may not be accurate.")
        LOG.warning("The versions are only updated after a `pip install -e .`")

    def validate_environment(
        self,
        *,
        all_packages: bool = False,
        on_difference: Literal["warn", "error", "ignore", "return"] = "warn",
        exempt_packages: list[str] | None = None,
    ) -> bool | str:
        """Validate environment of the checkpoint against the current environment.

        Parameters
        ----------
        all_packages : bool, optional
            Check all packages in the environment (True) or just anemoi's (False), by default False.
        on_difference : Literal['warn', 'error', 'ignore', 'return'], optional
            What to do on difference, by default "warn"
        exempt_packages : list[str], optional
            List of packages to exempt from the check, by default EXEMPT_PACKAGES

        Returns
        -------
        Union[bool, str]
            boolean if `on_difference` is not 'return', otherwise formatted text of the differences
            True if environment is valid, False otherwise

        Raises
        ------
        RuntimeError
            If found difference and `on_difference` is 'error'
        ValueError
            If `on_difference` is not 'warn' or 'error'
        """
        from anemoi.inference.provenance import validate_environment

        return validate_environment(
            self,
            all_packages=all_packages,
            on_difference=on_difference,
            exempt_packages=exempt_packages,
        )

    ###########################################################################

    def _get_datasets_full_paths(self) -> list:
        """Get the full paths of the datasets used in the training.

        Returns
        -------
        list
            The list of full paths.
        """
        # This is a temporary method to get the full paths of the datasets used in the training.
        # we need to review what goes in the dataset metadata.

        result = []

        def _find(x: Any) -> None:
            if isinstance(x, str):
                result.append(x)

            if isinstance(x, list):
                for y in x:
                    if isinstance(y, str):
                        result.append(y)
                    else:
                        _find(y)

            if isinstance(x, dict):
                if "dataset" in x and isinstance(x["dataset"], str):
                    result.append(x["dataset"])

                for k, v in x.items():
                    _find(v)

        dl = self._metadata.open_dataset_args(self.dataset_name)
        # open_dataset_args returns {"args": [...], "kwargs": {...}} or similar
        _find(dl.get("kwargs", dl) if isinstance(dl, dict) else dl)
        # Also check the args list for dataset path strings
        _find(dl.get("args", []) if isinstance(dl, dict) else [])
        return result

    def open_dataset(
        self,
        *,
        use_original_paths: bool | None = None,
        from_dataloader: str | None = None,
    ) -> tuple[Any, Any]:
        """Open the dataset.

        Parameters
        ----------
        use_original_paths : bool
            Whether to use the original paths.
        from_dataloader : str, optional
            The dataloader to use, by default None.

        Returns
        -------
        tuple
            The opened dataset and its arguments.
        """
        from anemoi.datasets import open_dataset
        from anemoi.utils.config import temporary_config

        if use_original_paths is not None:
            args, kwargs = self.open_dataset_args_kwargs(
                use_original_paths=use_original_paths, from_dataloader=from_dataloader
            )
            return open_dataset(*args, **kwargs)

        args, kwargs = self.open_dataset_args_kwargs(use_original_paths=True, from_dataloader=from_dataloader)

        #  Extract paths

        paths = []

        def _(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: _(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_(v) for v in x]

            if isinstance(x, str):
                if x.endswith(".zarr"):
                    paths.append(os.path.basename(x))
                    x = os.path.basename(x)
                    x = os.path.splitext(x)[0]

            return x

        args, kwargs = _((args, kwargs))

        with temporary_config(dict(datasets=dict(use_search_path_not_found=True))):
            return open_dataset(*args, **kwargs)

    def open_dataset_args_kwargs(
        self, *, use_original_paths: bool, from_dataloader: str | None = None
    ) -> tuple[Any, Any]:
        """Get the arguments and keyword arguments for opening the dataset.

        Parameters
        ----------
        use_original_paths : bool
            Whether to use the original paths.
        from_dataloader : str, optional
            The dataloader to use, by default None.

        Returns
        -------
        tuple
            The arguments and keyword arguments.
        """
        # Rebuild the full paths
        # Some older checkpoints may not have the full paths

        full_paths = self._get_datasets_full_paths()
        mapping = {}

        for path in full_paths:
            mapping[os.path.basename(path)] = path
            mapping[os.path.splitext(os.path.basename(path))[0]] = path

        def _fix(x: Any) -> Any:
            if isinstance(x, list):
                return [_fix(a) for a in x]

            if isinstance(x, dict):
                return {k: _fix(v) for k, v in x.items()}

            if isinstance(x, str):
                return mapping.get(x, x)

            return x

        if from_dataloader is not None:
            args, kwargs = (
                [],
                self._metadata.dataloader_config(from_dataloader, self.dataset_name),
            )
        else:
            arguments = self._metadata.open_dataset_args(self.dataset_name)
            args = arguments.get("args", [])
            kwargs = arguments.get("kwargs", {})

        args, kwargs = _fix([args, kwargs])

        # Remove keys that should not be passed to open_dataset()
        # See: https://github.com/ecmwf/anemoi-core/pull/756
        if isinstance(kwargs, dict):
            kwargs.pop("trajectory", None)

        if use_original_paths:
            return args, kwargs

        return _remove_full_paths(args), _remove_full_paths(kwargs)

    ###########################################################################
    # Not sure this belongs here
    # We need factories
    ###########################################################################

    def variable_categories(self) -> dict:
        """Get the categories of variables.

        Returns
        -------
        dict
            Mapping of variable name to sorted list of category strings.
        """
        if self._variables_categories is not None:
            return self._variables_categories

        result = self._metadata.variable_categories(self.dataset_name, per_variable=True)

        for name in self.variables:
            if name not in result:
                raise ValueError(f"Variable {name} has no category")

        self._variables_categories = frozendict(result)
        return self._variables_categories

    ###########################################################################
    # Supporting arrays
    ###########################################################################

    def load_supporting_array(self, name: str) -> FloatArray:
        """Load a supporting array by name.

        Parameters
        ----------
        name : str
            The name of the supporting array.

        Returns
        -------
        FloatArray
            The supporting array.

        Raises
        ------
        ValueError
            If the supporting array is not found.
        """
        if name not in self._supporting_arrays:
            LOG.error("No supporting array named `%s` found.", name)
            LOG.error("Supporting arrays found:")
            for names in self._supporting_arrays.keys():
                LOG.error("  %s", names)
            raise ValueError(f"Supporting array `{name}` not found")
        return self._supporting_arrays[name]

    @property
    def supporting_arrays(self) -> dict[str, FloatArray]:
        """Return the supporting arrays."""
        return self._supporting_arrays

    @property
    def latitudes(self) -> FloatArray | None:
        """Return the latitudes."""
        return self._supporting_arrays.get("latitudes")

    @property
    def longitudes(self) -> FloatArray | None:
        """Return the longitudes."""
        return self._supporting_arrays.get("longitudes")

    @property
    def grid_points_mask(self) -> FloatArray | None:
        """Return the grid points mask."""
        # Not yet available from pkg metadata; returns None until supported
        return None

    def provenance_training(self) -> dict[str, Any]:
        """Get the environmental configuration when trained.

        Returns
        -------
        dict
            The environmental configuration.
        """
        return self._metadata.provenance()

    def sources(self, path: str) -> list:
        """Get the sources from the metadata.

        Parameters
        ----------
        path : str
            The path to the sources.

        Returns
        -------
        list
            The list of sources.

        Raises
        ------
        ValueError
            If not all paths were fixed.
        """
        source_configs = self._metadata.sources(self.dataset_name)
        if not source_configs:
            return []

        import zipfile

        from anemoi.utils.checkpoints import load_supporting_arrays

        ###########################################################################
        # With older metadata, the zarr path are not stored in the metadata
        # we need to fix that
        ###########################################################################

        full_paths = self._get_datasets_full_paths()
        LOG.info(full_paths)
        n = 0

        def _fix(x: Any) -> Any:
            nonlocal n

            if isinstance(x, list):
                [_fix(a) for a in x]

            if isinstance(x, dict):
                if x.get("action", "").startswith("zarr"):
                    LOG.info(n, x)
                    path_val = full_paths[n]
                    n += 1
                    x["path"] = path_val

                {k: _fix(v) for k, v in x.items()}

        _fix(source_configs)

        if n != len(full_paths):
            raise ValueError("Not all paths were fixed")

        ###########################################################################

        # Get supporting array paths from the raw metadata dict
        raw_dict = self._metadata.to_dict()
        supporting_arrays_paths = raw_dict.get("supporting_arrays_paths", {})

        sources = []

        with zipfile.ZipFile(path, "r") as zipf:
            for i, source in enumerate(source_configs):
                entries = {name: supporting_arrays_paths[name] for name in source.get("supporting_arrays", [])}
                arrays = load_supporting_arrays(zipf, entries)

                name = source.get("name")
                if name is None:
                    name = f"source{i}"

                sources.append(
                    SourceMetadata(
                        self,
                        name,
                        source,
                        supporting_arrays=arrays,
                    )
                )

        return sources

    def print_variable_categories(self, print=LOG.info) -> None:
        """Print the variable categories for debugging purposes."""
        length = max(len(name) for name in self.variables)
        for name, categories in sorted(self.variable_categories().items()):
            print(f"   {name:{length}} => {', '.join(categories)}")

    ###########################################################################

    def patch(self, patch: dict) -> list[str]:
        """Patch the metadata with the given patch.

        Parameters
        ----------
        patch : dict
            The patch to apply.

        Returns
        -------
        list[str]
            Dotted paths of metadata keys that did not exist before and were created by
            the patch (the keys are still applied). Only the top-most new key of any
            newly-created subtree is reported. A non-empty list may simply be a deliberate
            addition, but it can also mean the patch does not match this checkpoint's
            metadata schema (e.g. a stale patch written for an older schema) and an
            intended update silently landed under a new key instead.
        """

        new_keys: list[str] = []

        # Get the raw dict from the pkg instance
        raw_dict = self._metadata.to_dict()

        def merge(
            main: dict[str, Any],
            patch: dict[str, Any],
            path: str = "",
            parent_is_new: bool = False,
        ) -> None:

            for k, v in patch.items():
                key_path = f"{path}.{k}" if path else k
                is_new = k not in main
                if is_new and not parent_is_new:
                    new_keys.append(key_path)
                if isinstance(v, dict) and isinstance(main.get(k, {}), dict):
                    if k not in main:
                        main[k] = {}
                    merge(main[k], v, key_path, parent_is_new or is_new)
                else:
                    main[k] = v

        merge(raw_dict, patch)

        # Reconstruct the pkg instance from the patched dict (no migration needed
        # since we are patching an already-migrated dict).
        self._metadata = PkgMetadata.from_dict(raw_dict, migrate=False)

        # Invalidate cached state
        self._variables_categories = None
        # Clear cached_property values that may have been computed from _metadata
        for attr in (
            "dataset_names",
            "task",
            "timestep",
            "multi_step_input",
            "multi_step_output",
            "variable_to_input_tensor_index",
            "variable_to_output_tensor_index",
            "input_tensor_index_to_variable",
            "output_tensor_index_to_variable",
            "variables_metadata",
            "accumulations",
            "model_computed_variables",
            "precision",
            "lagged",
            "typed_variables",
            "index_to_variable",
            "prognostic_variables",
            "number_of_grid_points",
            "number_of_input_features",
            "prognostic_output_mask",
            "prognostic_input_mask",
            "input_shape",
            "output_shape",
        ):
            self.__dict__.pop(attr, None)

        return new_keys


class SourceMetadata(Metadata):
    """An object that holds metadata of a source. It is only the `dataset` and `supporting_arrays` parts of the metadata.
    The rest is forwarded to the parent metadata object.
    """

    def __init__(
        self,
        parent: Metadata,
        name: str,
        source_dict: dict,
        supporting_arrays: dict | None = None,
    ):
        """Initialize the SourceMetadata object.

        Parameters
        ----------
        parent : Metadata
            The parent metadata object.
        name : str
            The name of the source.
        source_dict : dict
            The source metadata dictionary (the individual source entry).
        supporting_arrays : dict, optional
            The supporting arrays, by default None.
        """
        if supporting_arrays is None:
            supporting_arrays = {}
        # SourceMetadata delegates most operations to the parent; it only
        # overrides the supporting arrays and the dataset section.
        # We pass the parent's _metadata through so the base class is satisfied.
        super().__init__(parent._metadata, supporting_arrays, dataset_name=parent.dataset_name)
        self.parent = parent
        self.name = name

    @property
    def latitudes(self) -> FloatArray | None:
        """Return the latitudes."""
        return self._supporting_arrays.get(f"{self.name}/latitudes")

    @property
    def longitudes(self) -> FloatArray | None:
        """Return the longitudes."""
        return self._supporting_arrays.get(f"{self.name}/longitudes")

    @property
    def grid_points_mask(self) -> FloatArray | None:
        """Return the grid points mask."""
        for k, v in self._supporting_arrays.items():
            # TODO: This is a bit of a hack
            if k.startswith(f"{self.name}/") and "mask" in k:
                return v
        return None
