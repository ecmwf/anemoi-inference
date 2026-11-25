# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Inference-side view of the checkpoint metadata.

Checkpoint metadata is read by :mod:`anemoi.metadata`.  A
:class:`~anemoi.inference.checkpoint.Checkpoint` holds one
:class:`anemoi.metadata.Metadata` (``checkpoint.metadata``); everything that
is a plain read of the metadata (variables and indices, task timing, grid,
supporting arrays, raw sections) is read from it directly::

    m = checkpoint.metadata
    m.task.timestep()  # checkpoint level: the task
    m.dataset(name).data.variable_indices()  # per dataset: the data component
    m.dataset(name).arrays.latitudes  # per dataset: the supporting arrays
    m.sections.config  # the raw dumps

This module adds only what is inference-specific.  :class:`InferenceMetadata` is the thin wrapper of one dataset of a
:class:`anemoi.metadata.Metadata`: ``richer_batch`` (version >= 3.0),
tensor shapes, the coordinates of the model grid, MARS request builders, field namers, the
``multi_dataset`` flag, the environment package comparison, the
``select_variables`` extras (``has_mars_requests``) and the opening of the
training dataset.  It holds no cached value of its own that outlives its
:class:`anemoi.metadata.Metadata`: a new metadata gives new wrappers
(see :meth:`anemoi.inference.checkpoint.Checkpoint.replace_metadata`).

Plugins
-------
Input, output, processor and forcing classes keep the signature
``(context, metadata, ...)``.  They receive the ``Checkpoint`` as
``context.checkpoint`` (``context.checkpoint.metadata`` is the
:class:`anemoi.metadata.Metadata`) and ``metadata``, the object for the
dataset they work on.  ``metadata`` is an instance of
``checkpoint.metadata_base`` (a subclass of :class:`InferenceMetadata`, hence
a thin wrapper over the metadata); ``metadata.core`` is the
:class:`anemoi.metadata.Metadata`, ``metadata.dataset_view`` its
``DatasetView`` (``.data``, ``.arrays``) and ``metadata.dataset_name`` the
dataset name.  A ``metadata_base`` subclass is built with
``cls(core, dataset_name=name, number_of_grid_points=n)``.
"""

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import earthkit.data as ekd
import numpy as np
from anemoi.metadata import DatasetView
from anemoi.metadata import Metadata as CoreMetadata
from anemoi.utils.logs import OnceLogger
from anemoi.utils.provenance import gather_provenance_info
from earthkit.data.utils.dates import to_datetime
from packaging.version import InvalidVersion
from packaging.version import Version

from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray

if TYPE_CHECKING:
    from earthkit.data import FieldList

LOG = logging.getLogger(__name__)
LOG_ONCE = OnceLogger(LOG)

#: Categories accepted on the command line (``anemoi-inference retrieve``).
VARIABLE_CATEGORIES = {
    "computed",
    "forcing",
    "diagnostic",
    "prognostic",
    "constant",
    "accumulation",
}

#: First metadata version with the richer batch layout.
RICHER_BATCH_VERSION = Version("3.0")


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


def new_keys_of_patch(
    base: Mapping[str, Any],
    patch: Mapping[str, Any],
    path: str = "",
    parent_is_new: bool = False,
) -> list[str]:
    """Return the keys *patch* would create in *base* (``with_overrides`` does not report them).

    Parameters
    ----------
    base : Mapping[str, Any]
        The metadata the patch is applied to (not modified).
    patch : Mapping[str, Any]
        The patch.
    path : str, optional
        Dotted path of *base* within the full metadata, used in the result.
    parent_is_new : bool, optional
        Whether a parent of *base* is already new, so only the top-most new key is reported.

    Returns
    -------
    list[str]
        Dotted paths of the keys that do not exist in *base*.  Only the
        top-most new key of a newly created subtree is reported.  A non-empty
        list may be a deliberate addition, or a patch that does not match the
        schema of the checkpoint (a stale patch, a typo).
    """
    new_keys: list[str] = []
    for key, value in patch.items():
        key_path = f"{path}.{key}" if path else key
        is_new = key not in base
        if is_new and not parent_is_new:
            new_keys.append(key_path)
        if isinstance(value, Mapping):
            existing = base.get(key, {})
            if isinstance(existing, Mapping):
                new_keys.extend(new_keys_of_patch(existing, value, key_path, parent_is_new or is_new))
    return new_keys


class InferenceMetadata:
    """Thin inference-side wrapper of one dataset of a :class:`anemoi.metadata.Metadata`.

    Parameters
    ----------
    metadata : anemoi.metadata.Metadata
        The metadata of the checkpoint.
    dataset_name : str, optional
        The dataset to wrap; it may be omitted when the checkpoint has one dataset.
    number_of_grid_points : int, optional
        The explicit number of grid points of the checkpoint (set on
        :class:`~anemoi.inference.checkpoint.Checkpoint`, which passes it when it builds the
        wrappers); it wins over the length of the supporting arrays (an external graph
        defines its own grid).
    """

    def __init__(
        self,
        metadata: CoreMetadata,
        dataset_name: str | None = None,
        *,
        number_of_grid_points: int | None = None,
    ) -> None:
        self.core = metadata
        self.dataset_name: str = metadata.dataset(dataset_name).name
        self._explicit_number_of_grid_points = number_of_grid_points

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.dataset_name!r})"

    @cached_property
    def dataset_view(self) -> DatasetView:
        """The ``DatasetView`` (``.data``, ``.arrays``) of this dataset."""
        return self.core.dataset(self.dataset_name)

    @property
    def _data(self) -> Any:
        return self.dataset_view.data

    ###########################################################################
    # Checkpoint flags
    ###########################################################################

    @cached_property
    def multi_dataset(self) -> bool:
        """Whether the metadata has a ``metadata_inference.dataset_names`` (the multi-dataset layout)."""
        inference = self.core.sections.get("metadata_inference")
        return isinstance(inference, dict) and "dataset_names" in inference

    @cached_property
    def richer_batch(self) -> bool:
        """Whether the checkpoint was written with metadata version 3.0 or later."""
        try:
            return Version(self.core.version or "") >= RICHER_BATCH_VERSION
        except InvalidVersion:
            return False

    @property
    def data_type(self) -> Literal["gridded", "tabular"]:
        """The data type of the dataset (``"gridded"`` for checkpoints that record none)."""
        return self._data.data_type()

    ###########################################################################
    # Shapes and coordinates
    ###########################################################################

    @property
    def number_of_grid_points(self) -> int | None:
        """Number of grid points per field.

        The explicit count of the checkpoint (:attr:`anemoi.inference.checkpoint.Checkpoint.number_of_grid_points`)
        wins; else the length of ``grid_indices``, of ``latitudes``, or the grid size recorded in the metadata.
        """
        if self._explicit_number_of_grid_points is not None:
            return self._explicit_number_of_grid_points
        return self.dataset_view.number_of_grid_points

    def coordinates(self) -> tuple[FloatArray | None, FloatArray | None]:
        """Return the ``(latitudes, longitudes)`` of the grid points of the model.

        These are the supporting arrays, restricted to ``grid_indices`` when the checkpoint has them
        (deprecated; a warning is logged once).  An array the checkpoint does not have is ``None``.
        """
        arrays = self.dataset_view.arrays
        latitudes, longitudes = arrays.latitudes, arrays.longitudes
        grid_indices = arrays.grid_indices
        if grid_indices is not None:
            LOG_ONCE.warning(
                "Models trained with the 'grid_indices' option have been deprecated, and support for them will be removed in a future version. "
                "If your model uses 'grid_indices', please pin your inference version or retrain your model."
            )
            # `grid_indices` keep a subset of the grid points of the stored coordinates
            latitudes = latitudes[grid_indices] if latitudes is not None else None
            longitudes = longitudes[grid_indices] if longitudes is not None else None
        return latitudes, longitudes

    @cached_property
    def input_shape(self) -> tuple[int, int, int, int]:
        """Shape of the input tensor."""
        n_step_input = self.core.task.for_dataset(self.dataset_name).n_step_input
        n_variables = len(self._data.variable_indices())
        if self.multi_dataset and self.richer_batch:
            # richer-batch dropped the batch dimension and inserted ensemble=1
            return (n_step_input, 1, self.number_of_grid_points, n_variables)  # time, ensemble, grid, variables
        return (1, n_step_input, self.number_of_grid_points, n_variables)  # batch, time, grid, variables

    @cached_property
    def output_shape(self) -> tuple[int, int, int, int, int] | tuple[int, int, int, int]:
        """Shape of the output tensor."""
        n_variables = len(self._data.output_variable_indices())
        # When multistep_output was introduced, the metadata schema version was not bumped, so checkpoints
        # without it (older ones) are detected by the absence of the config entry, in `training` or `task`.
        config = self.core.sections.config
        has_multistep_output = any("multistep_output" in (config.get(part) or {}) for part in ("training", "task"))
        if self.multi_dataset and has_multistep_output:
            n_step_output = self.core.task.for_dataset(self.dataset_name).n_step_output
            if self.richer_batch:
                # richer-batch dropped the batch dimension
                return (n_step_output, 1, self.number_of_grid_points, n_variables)  # time, ensemble, grid, variables
            return (1, n_step_output, 1, self.number_of_grid_points, n_variables)  # batch, time, ensemble, grid, vars
        return (1, 1, self.number_of_grid_points, n_variables)

    ###########################################################################
    # Variables
    ###########################################################################

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
            Categories to include.
        exclude: List[str]
            Categories to exclude.
        has_mars_requests: bool
            If True, only include variables that have MARS requests.

        Returns
        -------
        List[str]
            The list of variables.
        """
        if include is not None and exclude is not None:
            overlap = {frozenset(c.split("+")) for c in include} & {frozenset(c.split("+")) for c in exclude}
            if overlap:
                raise ValueError(f"Include and exclude sets must not overlap {include} & {exclude}")

        result = self._data.select_variables(include=include, exclude=exclude)
        if has_mars_requests:
            variables_metadata = self._data.variables_metadata()
            result = [v for v in result if "mars" in variables_metadata.get(v, {})]
        return result

    def variables_mask(self, *, variables: list[str]) -> IntArray:
        """Return the input tensor indices of *variables*."""
        indices = self._data.variable_indices()
        return np.array([indices[v] for v in variables])

    def select_variables_and_masks(
        self, *, include: list[str] | None = None, exclude: list[str] | None = None
    ) -> tuple[list[str], IntArray]:
        """Select variables and return them with their input tensor indices."""
        variables = self.select_variables(include=include, exclude=exclude, has_mars_requests=False)
        return variables, self.variables_mask(variables=variables)

    ###########################################################################
    # Debugging
    ###########################################################################

    def print_indices(self, print=LOG.info) -> None:
        """Print the input and output tensor indices for debugging purposes."""
        for title, mapping in (
            ("Input indices", self._data.input_index_to_variable()),
            ("Output indices", self._data.output_index_to_variable()),
        ):
            print("")
            print(f"[{self.dataset_name}] {title}")
            for index, name in sorted(mapping.items()):
                print(f"   {index:3d} - {name}")
            if not mapping:
                print("   <empty>")

    def print_variable_categories(self, print=LOG.info) -> None:
        """Print the variable categories for debugging purposes."""
        categories = self._data.variable_categories(per_variable=True)
        length = max((len(name) for name in categories), default=0)
        for name, names in sorted(categories.items()):
            print(f"   {name:{length}} => {', '.join(names)}")

    ###########################################################################
    # Naming
    ###########################################################################

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

        param_levelist_to_name: dict[tuple[Any, Any], str] = {}
        ambiguous_keys: set[tuple[Any, Any]] = set()
        for name, variable in self._data.typed_variables().items():
            key = (variable.param, variable.level)
            if key in param_levelist_to_name and param_levelist_to_name[key] != name:
                ambiguous_keys.add(key)
            else:
                param_levelist_to_name[key] = name
        for key in ambiguous_keys:
            param_levelist_to_name.pop(key, None)

        def namer(field: ekd.Field, metadata: dict[str, Any]) -> str:
            param, levelist, levtype = (
                metadata.get("param"),
                metadata.get("levelist"),
                metadata.get("levtype"),
            )

            # Bug in eccodes that returns levelist for single level fields in GRIB2
            if levtype in ("sfc", "o2d"):
                levelist = None

            name = param_levelist_to_name.get((param, levelist))
            if name is not None:
                return name

            if levelist is None:
                return param

            return f"{param}_{levelist}"

        return namer

    ###########################################################################
    # Data retrieval
    ###########################################################################

    def mars_input_requests(self) -> Iterator[DataRequest]:
        """Generate MARS input requests.

        Returns
        -------
        Iterator[DataRequest]
            The MARS requests.
        """

        variables_metadata = self._data.variables_metadata()
        for variable in self.select_variables(include=["prognostic", "forcing"], exclude=["computed", "diagnostic"]):
            metadata = variables_metadata[variable]

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

        variables_metadata = self._data.variables_metadata()
        for variable in self.select_variables(
            include=["prognostic", "forcing"],
            exclude=["computed", "diagnostic"],
        ):
            metadata = variables_metadata[variable]

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

        result: list[DataRequest] = []

        DEFAULT_KEYS = ("class", "expver", "type", "stream", "levtype")
        DEFAULT_KEYS_AND_TIME = ("class", "expver", "type", "stream", "levtype", "time")

        # ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC
        # The split oper/scda is a bit special

        # CHANGE: Avoid duplicate GRIB values when training on forecasts by removing the time dependency in KEYS.
        # Set always_split_time=True to restore it.
        # TODO: revisit this code, can probably be removed
        KEYS = {("oper", "fc"): DEFAULT_KEYS, ("scda", "fc"): DEFAULT_KEYS}

        requests = defaultdict(list)
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

        result = []
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

        variables_metadata = self._data.variables_metadata()
        for variable in variables:
            if variable not in variables_metadata:
                raise ValueError(f"Variable {variable} not found in the metadata")

            if "mars" not in variables_metadata[variable]:
                raise ValueError(f"Variable {variable} has no MARS metadata")

            mars = variables_metadata[variable]["mars"].copy()

            for k in ("date", "hdate", "time", "valid_datetime", "variable"):
                mars.pop(k, None)

            yield mars

    ###########################################################################
    # Error reporting and environment
    ###########################################################################

    def provenance_training(self) -> dict[str, Any]:
        """Get the environmental configuration when trained.

        Returns
        -------
        dict
            The environmental configuration.
        """
        return dict(self.core.sections.get("provenance_training") or {})

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

        Compares the packages recorded in ``provenance_training`` with the
        installed ones (``anemoi.metadata`` only compares the Python version).

        Parameters
        ----------
        all_packages : bool, optional
            Check all packages in the environment (True) or just anemoi's (False), by default False.
        on_difference : Literal['warn', 'error', 'ignore', 'return'], optional
            What to do on difference, by default "warn".
        exempt_packages : list[str], optional
            List of packages to exempt from the check, by default EXEMPT_PACKAGES.

        Returns
        -------
        Union[bool, str]
            boolean if `on_difference` is not 'return', otherwise formatted text of the differences.
            True if environment is valid, False otherwise.

        Raises
        ------
        RuntimeError
            If found difference and `on_difference` is 'error'.
        ValueError
            If `on_difference` is not 'warn' or 'error'.
        """
        from anemoi.inference.provenance import validate_environment

        return validate_environment(
            self,
            all_packages=all_packages,
            on_difference=on_difference,
            exempt_packages=exempt_packages,
        )

    ###########################################################################
    # Training dataset
    ###########################################################################

    def _dataloader_dataset(self, partition: str = "training") -> dict[str, Any]:
        """Dataloader dataset configuration for the given partition."""
        return self._data.dataloader_config(partition)

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

        _find(self._dataloader_dataset("training").get("dataset"))
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
            args, kwargs = [], self._dataloader_dataset(from_dataloader)
        else:
            arguments = self._data.open_dataset_args()
            args, kwargs = arguments.get("args", []), arguments.get("kwargs", {})

        args, kwargs = _fix([args, kwargs])

        # Remove keys that should not be passed to open_dataset()
        # See: https://github.com/ecmwf/anemoi-core/pull/756
        kwargs.pop("trajectory", None)

        if use_original_paths:
            return args, kwargs

        return _remove_full_paths(args), _remove_full_paths(kwargs)
