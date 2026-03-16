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
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal

import deprecation
import earthkit.data as ekd
from anemoi.utils.checkpoints import load_metadata
from earthkit.data.utils.dates import to_datetime

from anemoi.inference._version import __version__
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date

from .metadata import Metadata
from .metadata import MetadataFactory
from .metadata import Variable

LOG = logging.getLogger(__name__)


def get_multi_dataset_metadata(metadata: dict, supporting_arrays: dict, base_class=Metadata) -> dict[str, Metadata]:
    """Metadata for all datasets in the checkpoint, as a mapping from dataset name to Metadata object.
    For legacy checkpoints, the dataset name defaults to `data`.
    """
    dataset_names = metadata.get("metadata_inference", {}).get("dataset_names", ["data"])

    return {
        dataset: MetadataFactory(metadata, supporting_arrays, dataset_name=dataset, base_class=base_class)
        for dataset in dataset_names
    }


def _download_huggingfacehub(huggingface_config: Any) -> str:
    """Download model from huggingface.

    Parameters
    ----------
    huggingface_config : dict or str
        Configuration for downloading from huggingface.

    Returns
    -------
    str
        Path to the downloaded model.
    """
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

    def __init__(
        self,
        source: str | Metadata | dict[Literal["huggingface"], str | dict],
        *,
        metadata_base: type[Metadata] = Metadata,
        patch_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Checkpoint.

        Parameters
        ----------
        source : str | Metadata | dict[Literal["huggingface"], str | dict]
            The source of the checkpoint as a path, a metadata object, or huggingface mapping.
        metadata_base : type[Metadata], optional
            The base class for the checkpoint's metadata object, useful for overriding task/runner-specific metadata.
        patch_metadata : dict[str, Any], optional
            Metadata to patch the checkpoint with, by default None.
        """
        self._source = source
        self.metadata_base = metadata_base
        self.patch_metadata = patch_metadata

    def __repr__(self) -> str:
        """Represent the Checkpoint as a string.

        Returns
        -------
        str
            String representation of the Checkpoint.
        """
        return f"Checkpoint({self.path})"

    @cached_property
    def path(self) -> str:
        """Get the path to the checkpoint."""
        import json

        try:
            path = json.loads(self._source)
        except Exception:
            path = self._source

        if isinstance(path, (Path, str)):
            return str(path)
        elif isinstance(path, dict):
            if "huggingface" in path:
                return _download_huggingfacehub(path["huggingface"])
            pass
        raise TypeError(f"Cannot parse model path: {path}. It must be a path or dict")

    @cached_property
    def _metadata(self) -> Metadata:
        """Get the metadata."""

        if isinstance(self._source, Metadata):
            return self._source

        # for multi-dataset checkpoints, we assume that attributes accessed via here
        # are shared across datasets and we can use the metadata of the first dataset.
        # things that need dataset-specific metadata should get their own MultiDatasetMetadata directly
        # TODO: may need to find a better way ¯\_(ツ)_/¯
        multi_metadata = self.multi_dataset_metadata
        result = multi_metadata[next(iter(multi_metadata))]

        if self.patch_metadata:
            # TODO: figure out multi-datasets patching
            LOG.warning("Patching metadata with %r", self.patch_metadata)
            result.patch(self.patch_metadata)

        return result

    @cached_property
    def _raw_metadata(self) -> tuple[dict, dict]:
        return load_metadata(self.path, supporting_arrays=True)

    @cached_property
    def multi_dataset(self) -> bool:
        """Check if the checkpoint is a multi-dataset checkpoint."""
        metadata, _ = self._raw_metadata
        return "metadata_inference" in metadata and "dataset_names" in metadata["metadata_inference"]

    @cached_property
    def multi_dataset_metadata(self) -> dict[str, Metadata]:
        """Metadata for all datasets in the checkpoint, as a mapping from dataset name to Metadata object.
        For legacy checkpoints, the dataset name defaults to `data`.
        """
        metadata, supporting_arrays = self._raw_metadata
        return get_multi_dataset_metadata(metadata, supporting_arrays, base_class=self.metadata_base)

    ###########################################################################
    # Forwards used by the runner
    # We do not want to expose the metadata object directly
    # We do not use `getattr` to avoid exposing all methods and make debugging
    # easier
    ###########################################################################

    @property
    def timestep(self) -> Any:
        """Get the timestep."""
        return self._metadata.timestep

    @property
    def input_explicit_times(self) -> Any:
        """Get the input explicit times from metadata."""
        return self._metadata.input_explicit_times

    @property
    def target_explicit_times(self) -> Any:
        """Get the target explicit times."""
        return self._metadata.target_explicit_times

    @property
    def data_frequency(self) -> Any:
        """Get the data frequency."""
        return self._metadata.data_frequency

    @property
    def precision(self) -> Any:
        """Get the precision."""
        return self._metadata.precision

    @property
    def variable_to_input_tensor_index(self) -> Any:
        """Get the variable to input tensor index."""
        # TODO: used by mock model, trace
        return self._metadata.variable_to_input_tensor_index

    @property
    def typed_variables(self) -> dict[str, Variable]:
        """Get the typed variables."""
        # TODO: used by checks and mock model
        return self._metadata.typed_variables

    @property
    @deprecation.deprecated(
        deprecated_in="0.6.4",
        removed_in="0.8.0",
        current_version=__version__,
        details="Use `select_variables` instead.",
    )
    def prognostic_variables(self) -> Any:
        """Get the prognostic variables."""
        # TODO: used by checks
        return self._metadata.prognostic_variables

    @property
    def output_tensor_index_to_variable(self) -> Any:
        """Get the output tensor index to variable."""
        # TODO: used by trace, mock model
        return self._metadata.output_tensor_index_to_variable

    @property
    def accumulations(self) -> Any:
        """Get the accumulations."""
        # TODO: used by accumulate post-processor
        return self._metadata.accumulations

    @cached_property
    def sources(self) -> list["SourceCheckpoint"]:
        """Get the sources."""
        return [SourceCheckpoint(self, _) for _ in self._metadata.sources(self.path)]

    def default_namer(self, *args: Any, **kwargs: Any) -> Callable[[ekd.Field, Any], str]:
        """Return a callable that can be used to name fields.

        Parameters
        ----------
        *args : Any
            Additional arguments.

        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Callable
            The namer that was used to create the training dataset.
        """
        # TODO: used by post-processors and multio output plugin
        return self._metadata.default_namer(*args, **kwargs)

    def report_error(self) -> None:
        """Report an error."""
        self._metadata.report_error()

    def validate_environment(
        self,
        *,
        all_packages: bool = False,
        on_difference: Literal["warn", "error", "ignore", "return"] = "warn",
        exempt_packages: list[str] | None = None,
    ) -> bool | str:
        """Validate the environment.

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
        """
        return self._metadata.validate_environment(
            all_packages=all_packages, on_difference=on_difference, exempt_packages=exempt_packages
        )

    def print_indices(self, print=LOG.info) -> None:
        """Print the indices."""
        # TODO: used by runner init
        return self._metadata.print_indices(print=print)

    ###########################################################################
    # Variable categories
    ###########################################################################
    # TODO: used by inspect, retrieve, mock model
    def print_variable_categories(self, print=LOG.info) -> None:
        """Print the variable categories."""
        return self._metadata.print_variable_categories(print=print)

    def variable_categories(self) -> Any:
        """Get the variable categories.

        Returns
        -------
        Any
            The variable categories.
        """
        return self._metadata.variable_categories()

    def select_variables(
        self, *, include: list[str] | None = None, exclude: list[str] | None = None, has_mars_requests: bool = True
    ) -> list[str]:
        """Get variables from input.

        Parameters
        ----------
        include : Optional[List[str]]
            Categories to include.
        exclude : Optional[List[str]]
            Categories to exclude.
        has_mars_requests: bool
            If True, only include variables that have MARS requests.

        Returns
        -------
        List[str]
            The selected variables.

        """
        return self._metadata.select_variables(include=include, exclude=exclude, has_mars_requests=has_mars_requests)

    ###########################################################################
    def load_supporting_array(self, name: str) -> Any:
        """Load a supporting array.

        Parameters
        ----------
        name : str
            The name of the supporting array.

        Returns
        -------
        Any
            The supporting array.
        """
        # TODO
        return self._metadata.load_supporting_array(name)

    @property
    def supporting_arrays(self) -> Any:
        """Get the supporting arrays."""
        return self._metadata.supporting_arrays

    ###########################################################################

    @cached_property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields."""
        return self._metadata.lagged

    @property
    def multi_step_input(self) -> int:
        """Get the multi-step input."""
        return self._metadata.multi_step_input

    @property
    def multi_step_output(self) -> int:
        """Get the multi-step output."""
        return self._metadata.multi_step_output

    ###########################################################################
    # Data retrieval
    ###########################################################################
    @property
    def grid(self) -> Any:
        """Get the grid."""
        return self._metadata.grid

    @property
    def area(self) -> Any:
        """Get the area."""
        return self._metadata.area

    def mars_by_levtype(self, levtype: str) -> Any:
        """Get MARS requests by level type.

        Parameters
        ----------
        levtype : str
            The level type.

        Returns
        -------
        Any
            The MARS requests by level type.
        """
        return self._metadata.mars_by_levtype(levtype)

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
        variables : List[str]
            The list of variables.
        dates : List[Any]
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
        KEYS = {("oper", "fc"): DEFAULT_KEYS, ("scda", "fc"): DEFAULT_KEYS}

        requests = defaultdict(list)
        for r in self._metadata.mars_requests(variables=variables):
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

    ###########################################################################
    # supporting arrays
    ###########################################################################

    @cached_property
    def _supporting_arrays(self) -> Any:
        """Get the supporting arrays."""
        return self._metadata._supporting_arrays

    @property
    def name(self) -> Any:
        """Get the name."""
        return self._metadata.dataset_name

    ###########################################################################
    # Misc
    ###########################################################################

    def provenance_training(self) -> Any:
        """Get the provenance of the training.

        Returns
        -------
        Any
            The provenance of the training.
        """
        return self._metadata.provenance_training()


class SourceCheckpoint(Checkpoint):
    """A checkpoint that represents a source."""

    def __init__(self, owner: Checkpoint, metadata: Any) -> None:
        """Initialize the SourceCheckpoint.

        Parameters
        ----------
        owner : Checkpoint
            The owner checkpoint.
        metadata : Any
            The metadata for the source checkpoint.
        """
        super().__init__(owner.path)
        self._owner = owner
        self._metadata = metadata

    def __repr__(self) -> str:
        """Represent the SourceCheckpoint as a string.

        Returns
        -------
        str
            String representation of the SourceCheckpoint.
        """
        return f"Source({self.name}@{self.path})"

    @property
    def operational_config(self) -> dict[str, Any]:
        LOG.warning("The `operational_config` property is deprecated.")
        return False
