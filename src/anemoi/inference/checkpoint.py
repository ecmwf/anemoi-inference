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
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional

from anemoi.utils.checkpoints import load_metadata
from earthkit.data.utils.dates import to_datetime

from .metadata import Metadata

LOG = logging.getLogger(__name__)


def _download_huggingfacehub(huggingface_config) -> str:
    """Download model from huggingface"""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Could not import `huggingface_hub`, please run `pip install huggingface_hub`.") from e

    if isinstance(huggingface_config, str):
        huggingface_config = {"repo_id": huggingface_config}

    if "filename" in huggingface_config:
        return str(hf_hub_download(**huggingface_config))

    repo_path = Path(snapshot_download(**huggingface_config))
    ckpt_files = list(repo_path.glob("*.ckpt"))

    if len(ckpt_files) == 1:
        return str(ckpt_files[0])
    else:
        raise ValueError(
            f"None or Multiple ckpt files found in repo, {ckpt_files}.\nCannot pick one to load, please specify `filename`."
        )


class Checkpoint:
    """Represents an inference checkpoint."""

    def __init__(self, path, *, patch_metadata=None):
        self._path = path
        self.patch_metadata = patch_metadata

    def __repr__(self):
        return f"Checkpoint({self.path})"

    @cached_property
    def path(self) -> str:
        import json

        try:
            path = json.loads(self._path)
        except Exception:
            path = self._path

        if isinstance(path, (Path, str)):
            return str(path)
        elif isinstance(path, dict):
            if "huggingface" in path:
                return _download_huggingfacehub(path["huggingface"])
            pass
        raise TypeError(f"Cannot parse model path: {path}. It must be a path or dict")

    @cached_property
    def _metadata(self):
        try:
            result = Metadata(*load_metadata(self.path, supporting_arrays=True))
        except Exception as e:
            LOG.warning("Version for not support `supporting_arrays` (%s)", e)
            result = Metadata(load_metadata(self.path))

        if self.patch_metadata:
            LOG.warning("Patching metadata with %r", self.patch_metadata)
            result.patch(self.patch_metadata)

        return result

    ###########################################################################
    # Forwards used by the runner
    # We do not want to expose the metadata object directly
    # We do not use `getattr` to avoid exposing all methods and make debugging
    # easier
    ###########################################################################

    @property
    def timestep(self):
        return self._metadata.timestep

    @property
    def precision(self):
        return self._metadata.precision

    @property
    def number_of_grid_points(self):
        return self._metadata.number_of_grid_points

    @property
    def number_of_input_features(self):
        return self._metadata.number_of_input_features

    @property
    def variable_to_input_tensor_index(self):
        return self._metadata.variable_to_input_tensor_index

    @property
    def model_computed_variables(self):
        return self._metadata.model_computed_variables

    @property
    def typed_variables(self):
        return self._metadata.typed_variables

    @property
    def diagnostic_variables(self):
        return self._metadata.diagnostic_variables

    @property
    def prognostic_variables(self):
        return self._metadata.prognostic_variables

    @property
    def prognostic_output_mask(self):
        return self._metadata.prognostic_output_mask

    @property
    def prognostic_input_mask(self):
        return self._metadata.prognostic_input_mask

    @property
    def output_tensor_index_to_variable(self):
        return self._metadata.output_tensor_index_to_variable

    @property
    def accumulations(self):
        return self._metadata.accumulations

    @property
    def latitudes(self):
        return self._metadata.latitudes

    @property
    def longitudes(self):
        return self._metadata.longitudes

    @property
    def grid_points_mask(self):
        return self._metadata.grid_points_mask

    @cached_property
    def sources(self):
        return [SourceCheckpoint(self, _) for _ in self._metadata.sources(self.path)]

    def default_namer(self, *args, **kwargs):
        """
        Return a callable that can be used to name fields.
        In that case, return the namer that was used to create the
        training dataset.
        """
        return self._metadata.default_namer(*args, **kwargs)

    def report_error(self):
        self._metadata.report_error()

    def validate_environment(
        self,
        *,
        all_packages: bool = False,
        on_difference: str = "warn",
        exempt_packages: Optional[list[str]] = None,
    ) -> bool:
        return self._metadata.validate_environment(
            all_packages=all_packages, on_difference=on_difference, exempt_packages=exempt_packages
        )

    def open_dataset_args_kwargs(self, *, use_original_paths, from_dataloader=None):
        return self._metadata.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=from_dataloader,
        )

    def constant_forcings_inputs(self, runner, input_state):
        return self._metadata.constant_forcings_inputs(runner, input_state)

    def dynamic_forcings_inputs(self, runner, input_state):
        return self._metadata.dynamic_forcings_inputs(runner, input_state)

    def boundary_forcings_inputs(self, runner, input_state):
        return self._metadata.boundary_forcings_inputs(runner, input_state)

    def name_fields(self, fields, namer=None):
        return self._metadata.name_fields(fields, namer=namer)

    def sort_by_name(self, fields, namer=None, *args, **kwargs):
        return self._metadata.sort_by_name(fields, namer=namer, *args, **kwargs)

    def print_indices(self):
        return self._metadata.print_indices()

    def variable_categories(self):
        return self._metadata.variable_categories()

    def load_supporting_array(self, name):
        return self._metadata.load_supporting_array(name)

    ###########################################################################

    @cached_property
    def lagged(self):
        """Return the list of timedelta for the `multi_step_input` fields."""
        result = list(range(0, self._metadata.multi_step_input))
        result = [-s * self._metadata.timestep for s in result]
        return sorted(result)

    @property
    def multi_step_input(self):
        return self._metadata.multi_step_input

    def print_variable_categories(self):
        return self._metadata.print_variable_categories()

    ###########################################################################
    # Data retrieval
    ###########################################################################

    def variables_from_input(self, *, include_forcings):
        return self._metadata.variables_from_input(include_forcings=include_forcings)

    @property
    def grid(self):
        return self._metadata.grid

    @property
    def area(self):
        return self._metadata.area

    def mars_by_levtype(self, levtype):
        return self._metadata.mars_by_levtype(levtype)

    def mars_requests(self, *, variables, dates, use_grib_paramid=False, **kwargs):
        from earthkit.data.utils.availability import Availability

        assert variables, "No variables provided"

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        assert dates, "No dates provided"

        result = []

        DEFAULT_KEYS = ("class", "expver", "type", "stream", "levtype")
        DEFAULT_KEYS_AND_TIME = ("class", "expver", "type", "stream", "levtype", "time")

        # The split oper/scda is a bit special
        KEYS = {("oper", "fc"): DEFAULT_KEYS_AND_TIME, ("scda", "fc"): DEFAULT_KEYS_AND_TIME}

        requests = defaultdict(list)

        for r in self._metadata.mars_requests(variables=variables, use_grib_paramid=use_grib_paramid):
            for date in dates:

                r = r.copy()

                base = date
                step = str(r.get("step", 0)).split("-")[-1]
                step = int(step)
                base = base - datetime.timedelta(hours=step)

                r["date"] = base.strftime("%Y-%m-%d")
                r["time"] = base.strftime("%H%M")

                r.update(kwargs)  # We do it here so that the Availability can use that information

                keys = KEYS.get((r.get("stream"), r.get("type")), DEFAULT_KEYS)
                key = tuple(r.get(k) for k in keys)

                # Special case because of oper/scda

                requests[key].append(r)

        result = []
        for reqs in requests.values():

            compressed = Availability(reqs)
            for r in compressed.iterate():
                for k, v in r.items():
                    if isinstance(v, (list, tuple)) and len(v) == 1:
                        r[k] = v[0]
                if r:
                    result.append(r)

        return result

    ###########################################################################
    # supporting arrays
    ###########################################################################

    @cached_property
    def _supporting_arrays(self):
        return self._metadata._supporting_arrays

    @property
    def name(self):
        return self._metadata.name

    ###########################################################################
    # Misc
    ###########################################################################

    def provenance_training(self):
        return self._metadata.provenance_training()


class SourceCheckpoint(Checkpoint):
    """A checkpoint that represents a source."""

    def __init__(self, owner, metadata):
        super().__init__(owner.path)
        self._owner = owner
        self._metadata = metadata

    def __repr__(self):
        return f"Source({self.name}@{self.path})"
