# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import warnings
from collections import defaultdict
from functools import cached_property

import numpy as np
from anemoi.transform.variables import Variable
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from .legacy import LegacyMixin
from .patch import PatchMixin

LOG = logging.getLogger(__name__)


class Metadata(PatchMixin, LegacyMixin):
    """An object that holds metadata of a checkpoint."""

    def __init__(self, metadata, verbose=False):
        self._metadata = DotDict(metadata)
        self._verbose = verbose

        # shortcuts
        self._indices = self._metadata.data_indices
        self._config_data = self._metadata.config.data
        self._config_training = self._metadata.config.training

    ###########################################################################
    # Inference
    ###########################################################################

    @cached_property
    def frequency(self):
        """Model time stepping frequency"""
        return to_timedelta(self._config_data.frequency)

    @cached_property
    def precision(self):
        """Return the precision of the model (bits per float)"""
        return self._config_training.precision

    def _make_indices_mapping(self, indices_from, indices_to):
        assert len(indices_from) == len(indices_to)
        return {i: j for i, j in zip(indices_from, indices_to)}

    @property
    def variable_to_input_tensor_index(self):
        """Return the mapping between variable name and input tensor index"""
        mapping = self._make_indices_mapping(
            self._indices.data.input.full,
            self._indices.model.input.full,
        )
        return {v: mapping[i] for i, v in enumerate(self.variables) if i in mapping}

    @cached_property
    def output_tensor_index_to_variable(self):
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self._indices.model.output.full,
            self._indices.data.output.full,
        )
        return {k: self.variables[v] for k, v in mapping.items()}

    @cached_property
    def number_of_grid_points(self):
        """Return the number of grid points per fields"""
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
        return [name for name, v in typed_variables.items() if v.is_computed_forcing]

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
        """
        Return the indices and names of the computed forcings that are not constant in time
        """

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

    # @cached_property
    # def input_computed_forcing_variables(self):
    #     forcings = self._config_data.forcing
    #     return [_ for _ in forcings if self.typed_variables[_].is_computed_forcing]

    ###########################################################################
    # Variables
    ###########################################################################

    @property
    def variables(self):
        """Return the variables as found in the training dataset"""
        return self._metadata.dataset.variables

    @cached_property
    def variables_metadata(self):
        """Return the variables and their metadata as found in the training dataset"""
        try:
            result = self._metadata.dataset.variables_metadata
            self._legacy_check_variables_metadata(result)
        except AttributeError:
            return self._legacy_variables_metadata()

        if "constant_fields" in self._metadata.dataset:
            for name in self._metadata.dataset.constant_fields:
                result[name]["constant_in_time"] = True

        return result

    @cached_property
    def diagnostic_variables(self):
        """Variables that are marked as diagnostic"""
        return [self.index_to_variable[i] for i in self._indices.data.input.diagnostic]

    @cached_property
    def index_to_variable(self):
        """Return a mapping from index to variable name"""
        return {i: v for i, v in enumerate(self.variables)}

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

    ###########################################################################
    # Default namer
    ###########################################################################

    def default_namer(self, *args, **kwargs):
        """
        Return a callable that can be used to name earthkit-data fields.
        In that case, return the namer that was used to create the
        training dataset.
        """

        assert len(args) == 0, args
        assert len(kwargs) == 0, kwargs

        def namer(field, metadata):
            warnings.warn("TEMPORARY CODE: Use the remapping in the metadata")
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

    def mars_requests(self, use_grib_paramid=False, **kwargs):
        """Return a list of MARS requests for the variables in the dataset"""

        from anemoi.utils.grib import shortname_to_paramid
        from earthkit.data.utils.availability import Availability

        keys = ("class", "expver", "type", "stream", "levtype")
        pop = ("date", "time")

        requests = defaultdict(list)

        for variable, metadata in self.variables_metadata.items():

            if "mars" not in metadata:
                continue

            if variable in self.diagnostic_variables:
                continue

            mars = metadata["mars"].copy()
            mars.update(kwargs)  # We do it here so that the Availability can use that information

            key = tuple(mars.get(k) for k in keys)
            for k in pop:
                metadata.pop(k, None)

            if use_grib_paramid and "param" in mars:
                mars["param"] = shortname_to_paramid(mars["param"])

            requests[key].append(mars)

        for reqs in requests.values():

            compressed = Availability(reqs)
            for r in compressed.iterate():
                for k, v in r.items():
                    if isinstance(v, (list, tuple)) and len(v) == 1:
                        r[k] = v[0]
                if r:
                    yield r

    ###########################################################################
    # Error reporting
    ###########################################################################

    def report_error(self):
        from anemoi.utils.provenance import gather_provenance_info

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

    ###########################################################################
    def open_dataset_args_kwargs(self):
        def _(x):
            if isinstance(x, dict):
                return {k: _(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_(v) for v in x]
            if isinstance(x, str):
                return os.path.splitext(os.path.basename(x))[0]

        return _(self._metadata.dataset.arguments.args), _(self._metadata.dataset.arguments.kwargs)
