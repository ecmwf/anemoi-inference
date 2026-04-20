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
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal

from anemoi.utils.checkpoints import load_metadata

from anemoi.inference.config.utils import multi_datasets_config

from .metadata import Metadata
from .metadata import MetadataFactory

LOG = logging.getLogger(__name__)


def get_multi_dataset_metadata(metadata: dict, supporting_arrays: dict, base_class=Metadata) -> dict[str, Metadata]:
    """Metadata objects for all datasets in the raw metadata, as a mapping from dataset name to Metadata object.
    For legacy checkpoints, the dataset name defaults to `data`
    """
    dataset_names = metadata.get("metadata_inference", {}).get("dataset_names", ["data"])

    return {
        dataset: MetadataFactory(
            metadata,
            supporting_arrays,
            dataset_name=dataset,
            base_class=base_class,
        )
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

        # for multi-dataset checkpoints, we assume that metadata attributes accessed via the checkpoint
        # are shared across datasets and we can use the metadata of the first dataset.
        # objects that need dataset-specific metadata should get their own MultiDatasetMetadata directly
        # via self.multi_dataset_metadata or get_multi_dataset_metadata()
        multi_metadata = self.multi_dataset_metadata
        result = multi_metadata[next(iter(multi_metadata))]

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
        For legacy checkpoints, the dataset name defaults to `data`
        """
        metadata, supporting_arrays = self._raw_metadata
        multi_metadata = get_multi_dataset_metadata(metadata, supporting_arrays, base_class=self.metadata_base)

        if self.patch_metadata:
            for dataset, metadata in multi_metadata.items():
                patch = multi_datasets_config(self.patch_metadata, dataset, list(multi_metadata.keys()), strict=False)
                LOG.warning(f"[{dataset}] Patching metadata with {patch}")
                metadata.patch(patch)

        return multi_metadata

    ###########################################################################
    # Forwards to the metadata
    # We assume that all attributes here are shared across datasets for multi-dataset checkpoints.
    ###########################################################################
    @property
    def timestep(self) -> Any:
        """Get the timestep."""
        return self._metadata.timestep

    @property
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

    ###########################################################################
    # Misc
    ###########################################################################
    @cached_property
    def sources(self) -> list["SourceCheckpoint"]:
        """Get the sources."""
        return [SourceCheckpoint(self, _) for _ in self._metadata.sources(self.path)]

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
        super().__init__(owner.path)
        self._owner = owner
        self._metadata = metadata

    def __repr__(self) -> str:
        return f"Source({self.path})"

    @property
    def operational_config(self) -> dict[str, Any]:
        LOG.warning("The `operational_config` property is deprecated.")
        return False
