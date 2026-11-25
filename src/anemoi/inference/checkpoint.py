# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The inference checkpoint.

A :class:`Checkpoint` holds the metadata of a checkpoint as one
:class:`anemoi.metadata.Metadata` (``checkpoint.metadata``, with the
``patch_metadata`` of the run configuration applied) and the objects the
inference code works with: one :class:`~anemoi.inference.metadata.InferenceMetadata`
per dataset (``checkpoint.multi_dataset_metadata``), built from ``checkpoint.metadata``
with the ``metadata_base`` class.  See :mod:`anemoi.inference.metadata` for what
plugins receive.

To change the metadata of a loaded checkpoint (an external graph defines its own
grid, for example) use :meth:`Checkpoint.replace_metadata`: it replaces the
metadata, drops everything derived from the old one and rebuilds the per-dataset
objects on first use.  Read values through the checkpoint (or through
``multi_dataset_metadata``) after the call; do not keep views of the old metadata.
"""

import datetime
import logging
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
from anemoi.metadata import Metadata as CoreMetadata
from anemoi.utils.checkpoints import load_metadata

from anemoi.inference.config.utils import multi_datasets_config

from .metadata import InferenceMetadata
from .metadata import new_keys_of_patch

LOG = logging.getLogger(__name__)

# Metadata keys whose creation by a `patch_metadata` patch is expected and must not trigger
# the new-key warning emitted when the patch is applied. E.g. `constant_fields` is frequently
# added through a patch even though it's not in the checkpoint metadata. TODO: find a better solution?
EXPECTED_PATCH_NEW_KEYS = {"constant_fields"}

# Everything `Checkpoint` caches that is derived from its metadata.
_DERIVED = (
    "metadata",
    "dataset_names",
    "multi_dataset_metadata",
    "multi_dataset",
    "richer_batch",
    "_metadata",
)


def get_multi_dataset_metadata(
    metadata: dict,
    supporting_arrays: dict,
    base_class: type[InferenceMetadata] = InferenceMetadata,
) -> dict[str, InferenceMetadata]:
    """Metadata objects for all datasets in the raw metadata, as a mapping from dataset name to Metadata object.
    For legacy checkpoints, the dataset name defaults to `data`
    """
    core = CoreMetadata.from_dict(metadata, supporting_arrays or None)
    return {name: base_class(core, dataset_name=name) for name in core.dataset_names}


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
        source: str | InferenceMetadata | CoreMetadata | dict[Literal["huggingface"], str | dict],
        *,
        metadata_base: type[InferenceMetadata] = InferenceMetadata,
        patch_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Checkpoint.

        Parameters
        ----------
        source : str | InferenceMetadata | anemoi.metadata.Metadata | dict[Literal["huggingface"], str | dict]
            The source of the checkpoint as a path, a metadata object, or huggingface mapping.
        metadata_base : type[InferenceMetadata], optional
            The base class for the checkpoint's metadata objects, useful for overriding task/runner-specific
            metadata. It is built with ``metadata_base(core, dataset_name=..., number_of_grid_points=...)``.
        patch_metadata : dict[str, Any], optional
            Metadata to patch the checkpoint with, by default None.
        """
        self._source = source
        self.metadata_base = metadata_base
        self.patch_metadata = patch_metadata
        self._number_of_grid_points: int | None = None

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

    ###########################################################################
    # The metadata
    ###########################################################################

    @cached_property
    def metadata(self) -> CoreMetadata:
        """The metadata of the checkpoint, with ``patch_metadata`` applied."""
        if isinstance(self._source, InferenceMetadata):
            return self._source.core
        if isinstance(self._source, CoreMetadata):
            core = self._source
        else:
            raw, arrays = load_metadata(self.path, supporting_arrays=True)
            core = CoreMetadata.from_dict(raw, arrays)
        return self._apply_patch_metadata(core)

    def _apply_patch_metadata(self, core: CoreMetadata) -> CoreMetadata:
        """Apply `patch_metadata` to *core*, once per distinct patch, warning about the keys it creates."""
        if not self.patch_metadata:
            return core

        names = core.dataset_names
        applied: list[Any] = []
        for dataset in names:
            patch = multi_datasets_config(self.patch_metadata, dataset, names, strict=False)
            if not patch or patch in applied:
                continue
            applied.append(patch)

            LOG.warning(f"[{dataset}] Patching metadata with {patch}")
            new_keys = new_keys_of_patch(core.to_dict(), patch)
            new_keys = [key for key in new_keys if key.rsplit(".", 1)[-1] not in EXPECTED_PATCH_NEW_KEYS]
            if new_keys:
                LOG.warning(
                    "[%s] The metadata patch added %d new key(s) that did not exist before: %s. "
                    "These keys were applied as given; if that was not intended, it usually means the "
                    "patch does not match this checkpoint's metadata schema.",
                    dataset,
                    len(new_keys),
                    ", ".join(new_keys),
                )
            core = core.with_overrides(patch)

        return core

    def _reset(self) -> None:
        """Drop every cached value derived from the metadata."""
        for name in _DERIVED:
            self.__dict__.pop(name, None)

    def replace_metadata(
        self,
        patch: Mapping[str, Any] | None = None,
        supporting_arrays: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        number_of_grid_points: int | None = None,
    ) -> None:
        """Replace the metadata of the checkpoint.

        The new metadata is the current one with *patch* deep-merged in and the
        arrays of *supporting_arrays* replaced or added
        (:meth:`anemoi.metadata.Metadata.with_overrides`).  Everything cached from the old
        metadata is dropped, so values read through the checkpoint afterwards are the new ones.

        Parameters
        ----------
        patch : Mapping[str, Any], optional
            Partial raw metadata to merge in (lists replace).
        supporting_arrays : Mapping[str, Mapping[str, numpy.ndarray]], optional
            Arrays to replace or add, ``{dataset_name: {array_name: array}}``.
        number_of_grid_points : int, optional
            An explicit number of grid points (for every dataset); it wins over the length of
            ``grid_indices`` or ``latitudes``. ``None`` clears a count set by a previous call.
        """
        core = self.metadata
        if patch or supporting_arrays:
            core = core.with_overrides(patch or {}, supporting_arrays)

        self._reset()
        self.__dict__["metadata"] = core
        self._number_of_grid_points = number_of_grid_points

    @property
    def number_of_grid_points(self) -> int | None:
        """The explicit number of grid points set by :meth:`replace_metadata`, or ``None`` (then the metadata gives it).

        The checkpoint owns this value: the per-dataset ``InferenceMetadata`` objects are built with it.
        """
        return self._number_of_grid_points

    @cached_property
    def dataset_names(self) -> list[str]:
        """The names of the datasets of the checkpoint (``["data"]`` for a legacy checkpoint)."""
        return self.metadata.dataset_names

    @cached_property
    def multi_dataset(self) -> bool:
        """Check if the checkpoint is a multi-dataset checkpoint."""
        return "dataset_names" in (self.metadata.sections.get("metadata_inference") or {})

    @cached_property
    def richer_batch(self) -> bool:
        """Whether the checkpoint was written with metadata version 3.0 or later."""
        return self._metadata.richer_batch

    @cached_property
    def multi_dataset_metadata(self) -> dict[str, InferenceMetadata]:
        """Metadata for all datasets in the checkpoint, as a mapping from dataset name to Metadata object.
        For legacy checkpoints, the dataset name defaults to `data`
        """
        core = self.metadata
        return {
            name: self.metadata_base(core, dataset_name=name, number_of_grid_points=self.number_of_grid_points)
            for name in core.dataset_names
        }

    @cached_property
    def _metadata(self) -> InferenceMetadata:
        """The metadata object of the first dataset."""
        # for multi-dataset checkpoints, we assume that metadata attributes accessed via the checkpoint
        # are shared across datasets and we can use the metadata of the first dataset.
        # objects that need dataset-specific metadata should get their own object
        # via self.multi_dataset_metadata
        source = self._source
        if isinstance(source, InferenceMetadata):
            if source.core is self.metadata:
                return source
            # the metadata was replaced: the wrapper given at construction is stale, rebuild it
            if source.dataset_name in self.multi_dataset_metadata:
                return self.multi_dataset_metadata[source.dataset_name]
        return next(iter(self.multi_dataset_metadata.values()))

    ###########################################################################
    # Forwards to the metadata
    # We assume that all attributes here are shared across datasets for multi-dataset checkpoints.
    ###########################################################################
    @property
    def _timing(self) -> Any:
        """The task timing of the first dataset (``metadata.task.for_dataset``)."""
        return self.metadata.task.for_dataset(self._metadata.dataset_name)

    def _timing_value(self, name: str, what: str) -> Any:
        """Return the timing value *name*, raising a ``ValueError`` naming *what* when the metadata does not give it."""
        value = getattr(self._timing, name)
        if value is None:
            raise ValueError(f"The checkpoint metadata does not give {what}")
        return value

    @property
    def task(self) -> str:
        """Return task, which describes the Runner."""
        return self.metadata.task.task_name

    @property
    def timestep(self) -> datetime.timedelta:
        """Get the timestep."""
        return self._timing_value("timestep_delta", "the model timestep")

    @property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields."""
        return self._timing_value("lagged", "the input offsets (lagged)")

    @property
    def multi_step_input(self) -> int:
        """Get the multi-step input."""
        return self._timing_value("n_step_input", "the number of input steps")

    @property
    def multi_step_output(self) -> int:
        """Get the multi-step output."""
        return self._timing_value("n_step_output", "the number of output steps")

    @property
    def input_explicit_times(self) -> list[int]:
        """Get the input explicit times from metadata."""
        return self._timing_value("input_explicit_times", "the input explicit times")

    @property
    def target_explicit_times(self) -> list[int]:
        """Get the target explicit times."""
        return self._timing_value("target_explicit_times", "the target explicit times")

    @property
    def output_offsets(self) -> list[datetime.timedelta]:
        """Return the list output offsets, i.e. the relative output times of a single forward."""
        return self._timing_value("output_offsets", "the output offsets")

    @property
    def rollout_shift(self) -> datetime.timedelta:
        """Get the rollout shift."""
        return self._timing_value("rollout_shift", "the rollout shift")

    @property
    def data_frequency(self) -> Any:
        """Get the data frequency."""
        return self._metadata.dataset_view.data.data_frequency()

    @property
    def precision(self) -> Any:
        """Get the precision."""
        return self.metadata.sections.precision

    ###########################################################################
    # Misc
    ###########################################################################
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
            What to do on difference, by default "warn".
        exempt_packages : list[str], optional
            List of packages to exempt from the check, by default EXEMPT_PACKAGES.

        Returns
        -------
        Union[bool, str]
            boolean if `on_difference` is not 'return', otherwise formatted text of the differences.
            True if environment is valid, False otherwise.
        """
        return self._metadata.validate_environment(
            all_packages=all_packages,
            on_difference=on_difference,
            exempt_packages=exempt_packages,
        )

    def provenance_training(self) -> Any:
        """Get the provenance of the training.

        Returns
        -------
        Any
            The provenance of the training.
        """
        return self._metadata.provenance_training()

    def update_metadata_from_zarr(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get new metadata and supporting array dictionaries from the original training dataset.
        Useful if the training dataset metadata has been patched/updated.
        Replaces the metadata of the checkpoint with the new one (``patch_metadata`` applied again)
        and returns the new ``(metadata, supporting_arrays)`` as stored in the checkpoint file
        (the arrays are keyed by dataset name for a multi-dataset checkpoint).
        """

        new_metadata, new_supporting_arrays = load_metadata(self.path, supporting_arrays=True)

        for dataset, metadata in self.multi_dataset_metadata.items():
            ds = metadata.open_dataset()
            ds_metadata, ds_supporting_arrays = ds.metadata(), ds.supporting_arrays()
            assert "sources" in ds_metadata

            if self.multi_dataset:
                new_metadata["dataset"][dataset] = ds_metadata
                new_supporting_arrays[dataset] = ds_supporting_arrays
            else:
                new_metadata["dataset"] = ds_metadata
                new_supporting_arrays = ds_supporting_arrays

        core = CoreMetadata.from_dict(new_metadata, new_supporting_arrays)
        self._reset()
        self.__dict__["metadata"] = self._apply_patch_metadata(core)
        return new_metadata, new_supporting_arrays
