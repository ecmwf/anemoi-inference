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
from anemoi.transform.variables import Variable
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.provenance import gather_provenance_info

from anemoi.inference.forcings import Forcings
from anemoi.inference.types import DataRequest
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from .legacy import LegacyMixin
from .patch import PatchMixin

if TYPE_CHECKING:
    from earthkit.data import FieldList

USE_LEGACY = True

LOG = logging.getLogger(__name__)


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


class Metadata(PatchMixin, LegacyMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata: dict[str, Any], supporting_arrays: dict[str, FloatArray] = {}):
        """Initialize the Metadata object.

        Parameters
        ----------
        metadata : dict
            The metadata dictionary.
        supporting_arrays : dict, optional
            The supporting arrays, by default {}.
        """
        self._metadata = DotDict(metadata)
        assert isinstance(supporting_arrays, dict)
        self._supporting_arrays = supporting_arrays

    @property
    def _indices(self) -> DotDict:
        """Return the data indices."""
        return self._metadata.data_indices

    @property
    def _config_data(self) -> DotDict:
        """Return the data configuration."""
        return self._config.data

    @property
    def _config_training(self) -> DotDict:
        """Return the training configuration."""
        return self._config.training

    @property
    def _config_model(self) -> DotDict:
        """Return the model configuration."""
        return self._config.model

    @property
    def _config(self) -> DotDict:
        """Return the configuration."""
        return self._metadata.config

    @property
    def target_explicit_times(self) -> Any:
        """Return the target explicit times from the training configuration."""
        return self._config_training.explicit_times.target

    @property
    def input_explicit_times(self) -> Any:
        """Return the input explicit times from the training configuration."""
        return self._config_training.explicit_times.input

    ###########################################################################
    # Debugging
    ###########################################################################

    def _print_indices(self, title: str, indices: dict[str, list[int]], naming: dict, skip: list[str] = []) -> None:
        """Print indices for debugging purposes.

        Parameters
        ----------
        title : str
            The title for the indices.
        indices : Dict[str, List[int]]
            The indices to print.
        naming : dict
            The naming convention for the indices.
        skip : set, optional
            The set of indices to skip, by default set().
        """
        LOG.info("")
        LOG.info("%s:", title)

        for k, v in sorted(indices.items()):
            if k in skip:
                continue
            for name, idx in sorted(v.items()):
                entry = f"{k}.{name}"
                if entry in skip:
                    continue

                LOG.info("   %s:", f"{k}.{name}")
                for n in idx:
                    LOG.info(f"     {n:3d} - %s", naming[k].get(n, "?"))
                if not idx:
                    LOG.info("     <empty>")

    def print_indices(self) -> None:
        """Print data and model indices for debugging purposes."""
        v = {i: v for i, v in enumerate(self.variables)}
        r = {v: k for k, v in self.variable_to_input_tensor_index.items()}
        s = self.output_tensor_index_to_variable

        self._print_indices("Data indices", self._indices.data, dict(input=v, output=v), skip=["output"])
        self._print_indices("Model indices", self._indices.model, dict(input=r, output=s, skip=["output.full"]))

    ###########################################################################
    # Inference
    ###########################################################################

    @cached_property
    def timestep(self) -> datetime.timedelta:
        """Model time stepping timestep."""
        # frequency = to_timedelta(self._config_data.frequency)
        timestep = to_timedelta(self._config_data.timestep)
        return timestep

    @cached_property
    def precision(self) -> str | int:
        """Return the precision of the model (bits per float)."""
        return self._config_training.precision

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

    @property
    def variable_to_input_tensor_index(self) -> frozendict:
        """Return the mapping between variable name and input tensor index."""
        mapping = self._make_indices_mapping(
            self._indices.data.input.full,
            self._indices.model.input.full,
        )

        return frozendict({v: mapping[i] for i, v in enumerate(self.variables) if i in mapping})

    @cached_property
    def output_tensor_index_to_variable(self) -> frozendict:
        """Return the mapping between output tensor index and variable name."""
        mapping = self._make_indices_mapping(
            self._indices.model.output.full,
            self._indices.data.output.full,
        )
        return frozendict({k: self.variables[v] for k, v in mapping.items()})

    @cached_property
    def number_of_grid_points(self) -> int:
        """Return the number of grid points per fields."""
        if "grid_indices" in self._supporting_arrays:
            return len(self.load_supporting_array("grid_indices"))
        try:
            return self._metadata.dataset.shape[-1]
        except AttributeError:
            if not USE_LEGACY:
                raise
            return self._legacy_number_of_grid_points()

    @cached_property
    def number_of_input_features(self) -> int:
        """Return the number of input features."""
        return len(self._indices.model.input.full)

    @cached_property
    def model_computed_variables(self) -> tuple:
        """The initial conditions variables that need to be computed and not retrieved."""
        typed_variables = self.typed_variables
        return tuple(name for name, v in typed_variables.items() if v.is_computed_forcing)

    @cached_property
    def multi_step_input(self) -> int:
        """Number of past steps needed for the initial conditions tensor."""
        return self._config_training.multistep_input

    @cached_property
    def prognostic_output_mask(self) -> IntArray:
        """Return the prognostic output mask."""
        return np.array(self._indices.model.output.prognostic)

    @cached_property
    def prognostic_input_mask(self) -> IntArray:
        """Return the prognostic input mask."""
        return np.array(self._indices.model.input.prognostic)

    @cached_property
    def computed_time_dependent_forcings(self) -> tuple[np.ndarray, list]:
        """Return the indices and names of the computed forcings that are not constant in time."""
        # Mapping between model and data indices
        mapping = self._make_indices_mapping(
            self._indices.model.input.full,
            self._indices.data.input.full,
        )

        # Mapping between model indices and variable names
        forcings_variables = {self.variables[mapping[i]]: i for i in self._indices.model.input.forcing}
        typed_variables = self.typed_variables

        # Filter out the computed forcings that are not constant in time
        indices = []
        variables = []
        for name, idx in sorted(forcings_variables.items(), key=lambda x: x[1]):
            v = typed_variables[name]
            if v.is_computed_forcing and not v.is_constant_in_time:
                indices.append(idx)
                variables.append(name)

        return np.array(indices), variables

    @cached_property
    def computed_constant_forcings(self) -> tuple[FloatArray, list[str]]:
        """Return the indices and names of the computed forcings that are  constant in time."""
        # Mapping between model and data indices
        mapping = self._make_indices_mapping(
            self._indices.model.input.full,
            self._indices.data.input.full,
        )

        # Mapping between model indices and variable names
        forcings_variables = {self.variables[mapping[i]]: i for i in self._indices.model.input.forcing}
        typed_variables = self.typed_variables

        # Filter out the computed forcings that are not constant in time
        indices = []
        variables = []
        for name, idx in sorted(forcings_variables.items(), key=lambda x: x[1]):
            v = typed_variables[name]
            if v.is_computed_forcing and v.is_constant_in_time:
                indices.append(idx)
                variables.append(name)

        return np.array(indices), variables

    ###########################################################################
    # Variables
    ###########################################################################

    @property
    def variables(self) -> tuple:
        """Return the variables as found in the training dataset."""
        return tuple(self._metadata.dataset.variables)

    @cached_property
    def variables_metadata(self) -> dict[str, Any]:
        """Return the variables and their metadata as found in the training dataset."""
        try:
            result = self._metadata.dataset.variables_metadata
            if USE_LEGACY:
                self._legacy_check_variables_metadata(result)
        except AttributeError:
            if not USE_LEGACY:
                raise
            result = self._legacy_variables_metadata()

        if "constant_fields" in self._metadata.dataset:
            for name in self._metadata.dataset.constant_fields:
                if name not in result:
                    continue
                result[name]["constant_in_time"] = True

        return result

    @cached_property
    def diagnostic_variables(self) -> list:
        """Variables that are marked as diagnostic."""
        return [self.index_to_variable[i] for i in self._indices.data.input.diagnostic]

    @cached_property
    def prognostic_variables(self) -> list:
        """Variables that are marked as prognostic."""
        return [self.index_to_variable[i] for i in self._indices.data.input.prognostic]

    @cached_property
    def index_to_variable(self) -> frozendict:
        """Return a mapping from index to variable name."""
        return frozendict({i: v for i, v in enumerate(self.variables)})

    @cached_property
    def typed_variables(self) -> dict[str, Variable]:
        """Returns a strongly typed variables."""
        result = {name: Variable.from_dict(name, self.variables_metadata[name]) for name in self.variables}

        if "cos_latitude" in result:
            assert result["cos_latitude"].is_computed_forcing
            assert result["cos_latitude"].is_constant_in_time

        if "cos_julian_day" in result:
            assert result["cos_julian_day"].is_computed_forcing
            assert not result["cos_julian_day"].is_constant_in_time

        return result

    @cached_property
    def accumulations(self) -> list:
        """Return the indices of the variables that are accumulations."""
        return [v.name for v in self.typed_variables.values() if v.is_accumulation]

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
    def _data_request(self) -> DataRequest:
        """Return the data request as encoded in the dataset."""
        try:
            return self._metadata.dataset.data_request
        except AttributeError:
            return self._legacy_data_request()

    @property
    def grid(self) -> str | None:
        """Return the grid information."""
        return self._data_request.get("grid")

    @property
    def area(self) -> str | None:
        """Return the area information."""
        return self._data_request.get("area")

    def variables_from_input(self, *, include_forcings: bool) -> list:
        """Get variables from input.

        Parameters
        ----------
        include_forcings : bool
            Whether to include forcings.

        Returns
        -------
        list
            The list of variables.
        """
        variable_categories = self.variable_categories()
        result = []
        for variable, metadata in self.variables_metadata.items():

            if "mars" not in metadata:
                continue

            if "forcing" in variable_categories[variable]:
                if not include_forcings:
                    continue

            if "computed" in variable_categories[variable]:
                continue

            if "diagnostic" in variable_categories[variable]:
                continue

            result.append(variable)

        return result

    def mars_input_requests(self) -> Iterator[DataRequest]:
        """Generate MARS input requests.

        Returns
        -------
        Iterator[DataRequest]
            The MARS requests.
        """
        variable_categories = self.variable_categories()
        for variable in self.variables_from_input(include_forcings=True):

            if "diagnostic" in variable_categories[variable]:
                continue

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
        variable_categories = self.variable_categories()

        params = set()
        levels = set()

        for variable in self.variables_from_input(include_forcings=True):

            if "diagnostic" in variable_categories[variable]:
                continue

            metadata = self.variables_metadata[variable]

            mars = metadata["mars"]
            if mars.get("levtype") != levtype:
                continue

            if "param" in mars:
                params.add(mars["param"])

            if "levelist" in mars:
                levels.add(mars["levelist"])

        return params, levels

    def mars_requests(self, *, variables: list[str]) -> Iterator[DataRequest]:
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
        provenance = self._metadata.provenance_training

        def _print(title: str, provenance: dict[str, Any]) -> None:
            LOG.info("")
            LOG.info("%s:", title)
            for package, git in sorted(provenance.get("git_versions", {}).items()):
                if package.startswith("anemoi."):
                    sha1 = git.get("git", {}).get("sha1", "unknown")
                    LOG.info(f"   {package:20}: {sha1}")

            for package, version in sorted(provenance.get("module_versions", {}).items()):
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

        _find(self._config.dataloader.training.dataset)
        return result

    def open_dataset(
        self, *, use_original_paths: bool | None = None, from_dataloader: str | None = None
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
            args, kwargs = [], self._metadata.config.dataloader[from_dataloader]
        else:
            args, kwargs = self._metadata.dataset.arguments.args, self._metadata.dataset.arguments.kwargs

        args, kwargs = _fix([args, kwargs])

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
            The categories of variables.
        """
        result = defaultdict(set)
        typed_variables = self.typed_variables

        variables_in_data_space = self.variables
        variables_in_model_space = self.output_tensor_index_to_variable

        for name in self._config.data.forcing:
            result[name].add("forcing")

        for idx in self._indices.data.input.diagnostic:
            name = variables_in_data_space[idx]
            result[name].add("diagnostic")

        # assert self._indices.model.input.prognostic == self._indices.model.output.prognostic
        for idx in self._indices.model.output.prognostic:
            name = variables_in_model_space[idx]
            result[name].add("prognostic")

        for name, v in typed_variables.items():
            if v.is_accumulation:
                result[name].add("accumulation")

            if v.is_constant_in_time:
                result[name].add("constant")

            if v.is_computed_forcing:
                result[name].add("computed")

        for name in self.variables:
            if name not in result:
                raise ValueError(f"Variable {name} has no category")

            result[name] = sorted(result[name])

        return result

    def constant_forcings_inputs(self, context: object, input_state: dict) -> list:
        """Get the constant forcings inputs.

        Parameters
        ----------
        context : object
            The context object.
        input_state : dict
            The input state.

        Returns
        -------
        list
            The list of constant forcings inputs.
        """
        # TODO: this does not belong here

        result = []

        provided_variables = set(input_state["fields"].keys())

        # This will manage the dynamic forcings that are computed
        forcing_mask, forcing_variables = self.computed_constant_forcings

        # Ingore provided variables
        new_forcing_mask = []
        new_forcing_variables = []

        for i, name in zip(forcing_mask, forcing_variables):
            if name not in provided_variables:
                new_forcing_mask.append(i)
                new_forcing_variables.append(name)

        LOG.info(
            "Computed constant forcings: before %s, after %s",
            forcing_variables,
            new_forcing_variables,
        )

        forcing_mask = np.array(new_forcing_mask)
        forcing_variables = new_forcing_variables

        if len(forcing_mask) > 0:
            result.extend(
                context.create_constant_computed_forcings(
                    forcing_variables,
                    forcing_mask,
                )
            )

        remaining = (
            set(self._config.data.forcing)
            - set(self.model_computed_variables)
            - set(forcing_variables)
            - provided_variables
        )
        if not remaining:
            return result

        LOG.info("Remaining forcings: %s", remaining)

        # We need the mask of the remaining variable in the model.input space

        mapping = self._make_indices_mapping(
            self._indices.data.input.full,
            self._indices.model.input.full,
        )

        remaining = sorted((mapping[self.variables.index(name)], name) for name in remaining)

        LOG.info("Will get the following from MARS for now: %s", remaining)

        remaining_mask = [i for i, _ in remaining]
        remaining = [name for _, name in remaining]

        result.extend(
            context.create_constant_coupled_forcings(
                remaining,
                remaining_mask,
            )
        )

        return result

    def dynamic_forcings_inputs(self, context: object, input_state: State) -> list[Forcings]:
        """Get the dynamic forcings inputs.

        Parameters
        ----------
        context : object
            The context object.
        input_state : State
            The input state.

        Returns
        -------
        list
            The list of dynamic forcings inputs.
        """
        result = []

        # This will manage the dynamic forcings that are computed
        forcing_mask, forcing_variables = self.computed_time_dependent_forcings
        if len(forcing_mask) > 0:
            result.extend(
                context.create_dynamic_computed_forcings(
                    forcing_variables,
                    forcing_mask,
                )
            )

        remaining = (
            set(self._config.data.forcing)
            - set(self.model_computed_variables)
            - {name for name, v in self.typed_variables.items() if v.is_constant_in_time}
        )
        if not remaining:
            return result

        LOG.info("Remaining forcings: %s", remaining)

        # We need the mask of the remaining variable in the model.input space

        mapping = self._make_indices_mapping(
            self._indices.data.input.full,
            self._indices.model.input.full,
        )

        remaining = sorted((mapping[self.variables.index(name)], name) for name in remaining)

        LOG.info("Will get the following from `forcings.dynamic`: %s", remaining)

        remaining_mask = [i for i, _ in remaining]
        remaining = [name for _, name in remaining]

        result.extend(
            context.create_dynamic_coupled_forcings(
                remaining,
                remaining_mask,
            )
        )
        return result

    def boundary_forcings_inputs(self, context: object, input_state: dict) -> list:
        """Get the boundary forcings inputs.

        Parameters
        ----------
        context : object
            The context object.
        input_state : dict
            The input state.

        Returns
        -------
        list
            The list of boundary forcings inputs.
        """
        if "output_mask" not in self._supporting_arrays:
            return []

        return context.create_boundary_forcings(
            self.prognostic_variables,
            self.prognostic_input_mask,
        )

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
        # TODO
        return None

    def provenance_training(self) -> dict[str, Any]:
        """Get the environmental configuration when trained.

        Returns
        -------
        dict
            The environmental configuration.
        """
        return dict(self._metadata.get("provenance_training", {}))

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
        if "sources" not in self._metadata.dataset:
            return []

        import zipfile

        from anemoi.utils.checkpoints import load_supporting_arrays

        ###########################################################################
        # With older metadata, the zarr path are not stored in the metadata
        # we need to fix that
        ###########################################################################

        full_paths = self._get_datasets_full_paths()
        print(full_paths)
        n = 0

        def _fix(x: Any) -> Any:
            nonlocal n

            if isinstance(x, list):
                [_fix(a) for a in x]

            if isinstance(x, dict):
                if x.get("action", "").startswith("zarr"):
                    print(n, x)
                    path = full_paths[n]
                    n += 1
                    x["path"] = path

                {k: _fix(v) for k, v in x.items()}

        _fix(self._metadata.dataset.sources)

        if n != len(full_paths):
            raise ValueError("Not all paths were fixed")

        ###########################################################################

        sources = []

        with zipfile.ZipFile(path, "r") as zipf:
            for i, source in enumerate(self._metadata.dataset.get("sources", [])):
                entries = {
                    name: self._metadata.supporting_arrays_paths[name] for name in source.get("supporting_arrays", [])
                }
                arrays = load_supporting_arrays(zipf, entries)

                name = source.get("name")
                if name is None:
                    name = f"source{i}"

                sources.append(
                    SourceMetadata(
                        self,
                        name,
                        dict(dataset=source),
                        supporting_arrays=arrays,
                    )
                )

        return sources

    def print_variable_categories(self) -> None:
        """Print the variable categories for debugging purposes."""
        length = max(len(name) for name in self.variables)
        for name, categories in sorted(self.variable_categories().items()):
            LOG.info(f"   {name:{length}} => {', '.join(categories)}")

    ###########################################################################

    def patch(self, patch: dict) -> None:
        """Patch the metadata with the given patch.

        Parameters
        ----------
        patch : dict
            The patch to apply.
        """

        def merge(main: dict[str, Any], patch: dict[str, Any]) -> None:

            for k, v in patch.items():
                if isinstance(v, dict):
                    if k not in main:
                        main[k] = {}
                    merge(main[k], v)
                else:
                    main[k] = v

        merge(self._metadata, patch)


class SourceMetadata(Metadata):
    """An object that holds metadata of a source. It is only the `dataset` and `supporting_arrays` parts of the metadata.
    The rest is forwarded to the parent metadata object.
    """

    def __init__(self, parent: Metadata, name: str, metadata: dict, supporting_arrays: dict = {}):
        """Initialize the SourceMetadata object.

        Parameters
        ----------
        parent : Metadata
            The parent metadata object.
        name : str
            The name of the source.
        metadata : dict
            The metadata dictionary.
        supporting_arrays : dict, optional
            The supporting arrays, by default {}.
        """
        super().__init__(metadata, supporting_arrays)
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

    ###########################################################################
    # Forward to parent metadata
    ###########################################################################

    @property
    def _config_training(self) -> DotDict:
        """Return the training configuration."""
        return self.parent._config_training

    @property
    def _config_data(self) -> DotDict:
        """Return the data configuration."""
        return self.parent._config_data

    @property
    def _indices(self) -> DotDict:
        """Return the indices."""
        return self.parent._indices

    @property
    def _config(self) -> DotDict:
        """Return the configuration."""
        return self.parent._config

    ###########################################################################
