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
import warnings
from collections import defaultdict
from functools import cached_property
from types import MappingProxyType as frozendict
from typing import Literal
from typing import Optional

import numpy as np
from anemoi.transform.variables import Variable
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.provenance import gather_provenance_info

from .legacy import LegacyMixin
from .patch import PatchMixin

LOG = logging.getLogger(__name__)


def _remove_full_paths(x):
    if isinstance(x, dict):
        return {k: _remove_full_paths(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_remove_full_paths(v) for v in x]
    if isinstance(x, str):
        return os.path.splitext(os.path.basename(x))[0]
    return x


class Metadata(PatchMixin, LegacyMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata, supporting_arrays={}):
        self._metadata = DotDict(metadata)
        assert isinstance(supporting_arrays, dict)
        self._supporting_arrays = supporting_arrays

    @property
    def _indices(self):
        return self._metadata.data_indices

    @property
    def _config_data(self):
        return self._config.data

    @property
    def _config_training(self):
        return self._config.training

    @property
    def _config_model(self):
        return self._config.model

    @property
    def _config(self):
        return self._metadata.config

    ###########################################################################
    # Debugging
    ###########################################################################

    def _print_indices(self, title, indices, naming, skip=set()):
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

    def print_indices(self):

        v = {i: v for i, v in enumerate(self.variables)}
        r = {v: k for k, v in self.variable_to_input_tensor_index.items()}
        s = self.output_tensor_index_to_variable

        self._print_indices("Data indices", self._indices.data, dict(input=v, output=v), skip=["output"])
        self._print_indices("Model indices", self._indices.model, dict(input=r, output=s, skip=["output.full"]))

    ###########################################################################
    # Inference
    ###########################################################################

    @cached_property
    def timestep(self):
        """Model time stepping timestep"""
        # frequency = to_timedelta(self._config_data.frequency)
        timestep = to_timedelta(self._config_data.timestep)
        return timestep

    @cached_property
    def precision(self):
        """Return the precision of the model (bits per float)"""
        return self._config_training.precision

    def _make_indices_mapping(self, indices_from, indices_to):
        assert len(indices_from) == len(indices_to)
        return frozendict({i: j for i, j in zip(indices_from, indices_to)})

    @property
    def variable_to_input_tensor_index(self):
        """Return the mapping between variable name and input tensor index"""
        mapping = self._make_indices_mapping(
            self._indices.data.input.full,
            self._indices.model.input.full,
        )

        return frozendict({v: mapping[i] for i, v in enumerate(self.variables) if i in mapping})

    @cached_property
    def output_tensor_index_to_variable(self):
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self._indices.model.output.full,
            self._indices.data.output.full,
        )
        return frozendict({k: self.variables[v] for k, v in mapping.items()})

    @cached_property
    def number_of_grid_points(self):
        """Return the number of grid points per fields"""
        if "grid_indices" in self._supporting_arrays:
            return len(self.load_supporting_array("grid_indices"))
        try:
            return self._metadata.dataset.shape[-1]
        except AttributeError:
            return self._legacy_number_of_grid_points()

    @cached_property
    def number_of_input_features(self):
        """Return the number of input features"""
        return len(self._indices.model.input.full)

    @cached_property
    def model_computed_variables(self):
        """The initial conditions variables that need to be computed and not retrieved"""
        typed_variables = self.typed_variables
        return tuple(name for name, v in typed_variables.items() if v.is_computed_forcing)

    @cached_property
    def multi_step_input(self):
        """Number of past steps needed for the initial conditions tensor"""
        return self._config_training.multistep_input

    @cached_property
    def prognostic_output_mask(self):
        return np.array(self._indices.model.output.prognostic)

    @cached_property
    def prognostic_input_mask(self):
        return np.array(self._indices.model.input.prognostic)

    @cached_property
    def computed_time_dependent_forcings(self):
        """Return the indices and names of the computed forcings that are not constant in time"""

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
    def computed_constant_forcings(self):
        """Return the indices and names of the computed forcings that are  constant in time"""

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
    def variables(self):
        """Return the variables as found in the training dataset"""
        return tuple(self._metadata.dataset.variables)

    @cached_property
    def variables_metadata(self):
        """Return the variables and their metadata as found in the training dataset"""
        try:
            result = self._metadata.dataset.variables_metadata
            self._legacy_check_variables_metadata(result)
        except AttributeError:
            result = self._legacy_variables_metadata()

        if "constant_fields" in self._metadata.dataset:
            for name in self._metadata.dataset.constant_fields:
                result[name]["constant_in_time"] = True

        return result

    @cached_property
    def diagnostic_variables(self):
        """Variables that are marked as diagnostic"""
        return [self.index_to_variable[i] for i in self._indices.data.input.diagnostic]

    @cached_property
    def prognostic_variables(self):
        """Variables that are marked as prognostic"""
        return [self.index_to_variable[i] for i in self._indices.data.input.prognostic]

    @cached_property
    def index_to_variable(self):
        """Return a mapping from index to variable name"""
        return frozendict({i: v for i, v in enumerate(self.variables)})

    @cached_property
    def typed_variables(self):
        """Returns a strongly typed variables"""
        result = {name: Variable.from_dict(name, self.variables_metadata[name]) for name in self.variables}

        if "cos_latitude" in result:
            assert result["cos_latitude"].is_computed_forcing
            assert result["cos_latitude"].is_constant_in_time

        if "cos_julian_day" in result:
            assert result["cos_julian_day"].is_computed_forcing
            assert not result["cos_julian_day"].is_constant_in_time

        return result

    @cached_property
    def accumulations(self):
        """Return the indices of the variables that are accumulations"""
        return [v.name for v in self.typed_variables.values() if v.is_accumulation]

    def name_fields(self, fields, namer=None):
        from earthkit.data.indexing.fieldlist import FieldArray

        if namer is None:
            namer = self.default_namer()

        def _name(field, _, original_metadata):
            return namer(field, original_metadata)

        return FieldArray([f.copy(name=_name) for f in fields])

    def sort_by_name(self, fields, namer=None, *args, **kwargs):
        fields = self.name_fields(fields, namer=namer)
        return fields.order_by("name", *args, **kwargs)

    ###########################################################################
    # Default namer
    ###########################################################################

    def default_namer(self, *args, **kwargs):
        """Return a callable that can be used to name earthkit-data fields.
        In that case, return the namer that was used to create the
        training dataset.
        """

        assert len(args) == 0, args
        assert len(kwargs) == 0, kwargs

        def namer(field, metadata):
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
    def _data_request(self):
        """Return the data request as encoded in the dataset"""
        try:
            return self._metadata.dataset.data_request
        except AttributeError:
            return self._legacy_data_request()

    @property
    def grid(self):
        return self._data_request.get("grid")

    @property
    def area(self):
        return self._data_request.get("area")

    def variables_from_input(self, *, include_forcings):
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

    def mars_input_requests(self):
        variable_categories = self.variable_categories()
        for variable in self.variables_from_input(include_forcings=True):

            if "diagnostic" in variable_categories[variable]:
                continue

            metadata = self.variables_metadata[variable]

            yield metadata["mars"].copy()

    def mars_by_levtype(self, levtype):
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

    def mars_requests(self, *, variables, use_grib_paramid=False):
        """Return a list of MARS requests for the variables in the dataset"""

        from anemoi.utils.grib import shortname_to_paramid

        if len(variables) == 0:
            raise ValueError("No variables requested")

        for variable in variables:

            if variable not in self.variables_metadata:
                raise ValueError(f"Variable {variable} not found in the metadata")

            if "mars" not in self.variables_metadata[variable]:
                raise ValueError(f"Variable {variable} has no MARS metadata")

            mars = self.variables_metadata[variable]["mars"].copy()

            for k in ("date", "time"):
                mars.pop(k, None)

            if use_grib_paramid and "param" in mars:
                mars["param"] = shortname_to_paramid(mars["param"])

            yield mars

    ###########################################################################
    # Error reporting
    ###########################################################################

    def report_error(self):

        provenance = self._metadata.provenance_training

        def _print(title, provenance):
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
        on_difference: Literal["warn", "error", "ignore"] = "warn",
        exempt_packages: Optional[list[str]] = None,
    ) -> bool:
        """
        Validate environment of the checkpoint against the current environment.

        Parameters
        ----------
        all_packages : bool, optional
            Check all packages in environment or just `anemoi`'s, by default False
        on_difference : Literal['warn', 'error', 'ignore'], optional
            What to do on difference, by default "warn"
        exempt_packages : list[str], optional
            List of packages to exempt from the check, by default EXEMPT_PACKAGES

        Returns
        -------
        bool
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
            self, all_packages=all_packages, on_difference=on_difference, exempt_packages=exempt_packages
        )

    ###########################################################################

    def _get_datasets_full_paths(self):
        """This is a temporary method to get the full paths of the datasets used in the training.
        we need to review what goes in the dataset metadata
        """

        result = []

        def _find(x):

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

    def open_dataset_args_kwargs(self, *, use_original_paths, from_dataloader=None):

        # Rebuild the full paths
        # Some older checkpoints may not have the full paths

        full_paths = self._get_datasets_full_paths()
        mapping = {}

        for path in full_paths:
            mapping[os.path.basename(path)] = path
            mapping[os.path.splitext(os.path.basename(path))[0]] = path

        def _fix(x):
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

    def variable_categories(self):
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

    def constant_forcings_inputs(self, context, input_state):
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
            result.append(
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

        forcing = context.create_constant_coupled_forcings(remaining, remaining_mask)

        if forcing is not None:
            # SimpleRunner does not support dynamic forcings
            result.append(forcing)

        return result

    def dynamic_forcings_inputs(self, context, input_state):

        result = []

        # This will manage the dynamic forcings that are computed
        forcing_mask, forcing_variables = self.computed_time_dependent_forcings
        if len(forcing_mask) > 0:
            result.append(
                context.create_dynamic_computed_forcings(
                    forcing_variables,
                    forcing_mask,
                )
            )

        remaining = (
            set(self._config.data.forcing)
            - set(self.model_computed_variables)
            - set([name for name, v in self.typed_variables.items() if v.is_constant_in_time])
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

        forcing = context.create_dynamic_coupled_forcings(remaining, remaining_mask)

        if forcing is not None:
            # SimpleRunner does not support dynamic forcings
            result.append(forcing)

        return result

    def boundary_forcings_inputs(self, context, input_state):

        result = []

        if "output_mask" in self._supporting_arrays:
            result.append(
                context.create_boundary_forcings(
                    self.prognostic_variables,
                    self.prognostic_input_mask,
                )
            )
        return result

    ###########################################################################
    # Supporting arrays
    ###########################################################################

    def load_supporting_array(self, name):
        if name not in self._supporting_arrays:
            LOG.error("No supporting array named `%s` found.", name)
            LOG.error("Supporting arrays found:")
            for names in self._supporting_arrays.keys():
                LOG.error("  %s", names)
            raise ValueError(f"Supporting array `{name}` not found")
        return self._supporting_arrays[name]

    @property
    def latitudes(self):
        return self._supporting_arrays.get("latitudes")

    @property
    def longitudes(self):
        return self._supporting_arrays.get("longitudes")

    @property
    def grid_points_mask(self):
        # TODO
        return None

    def provenance_training(self):
        """Environmental Configuration when trained"""
        return dict(self._metadata.get("provenance_training", {}))

    def sources(self, path):

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

        def _fix(x):
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

    def print_variable_categories(self):
        length = max(len(name) for name in self.variables)
        for name, categories in sorted(self.variable_categories().items()):
            LOG.info(f"   {name:{length}} => {', '.join(categories)}")

    ###########################################################################

    def patch(self, patch):
        """Patch the metadata with the given patch"""

        def merge(main, patch):

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

    def __init__(self, parent, name, metadata, supporting_arrays={}):
        super().__init__(metadata, supporting_arrays)
        self.parent = parent
        self.name = name

    @property
    def latitudes(self):
        return self._supporting_arrays.get(f"{self.name}/latitudes")

    @property
    def longitudes(self):
        return self._supporting_arrays.get(f"{self.name}/longitudes")

    @property
    def grid_points_mask(self):
        for k, v in self._supporting_arrays.items():
            # TODO: This is a bit of a hack
            if k.startswith(f"{self.name}/") and "mask" in k:
                return v
        return None

    ###########################################################################
    # Forward to parent metadata
    ###########################################################################

    @property
    def _config_training(self):
        return self.parent._config_training

    @property
    def _config_data(self):
        return self.parent._config_data

    @property
    def _indices(self):
        return self.parent._indices

    @property
    def _config(self):
        return self.parent._config

    ###########################################################################
