# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the sanitise command's dataset-metadata handling (issue #618)."""

from argparse import Namespace

import pytest

from anemoi.inference.commands.sanitise import SanitiseCmd
from anemoi.inference.commands.sanitise import _iter_dataset_metadatas

# A flat (legacy single-dataset) metadata dict: the dataset dict IS metadata["dataset"].
# It carries an absolute path so that `sanitise()` changes it, forcing the patch branch.
FLAT_METADATA = {
    "version": "2.0",
    "dataset": {
        "sources": [{"type": "zarr", "dataset": "/absolute/path/to/data.zarr"}],
        "variables": ["2t"],
    },
}

# A named (metadata_inference.dataset_names set) metadata dict, as written by current
# anemoi-training runs: metadata["dataset"] maps each dataset name to its metadata.
NAMED_METADATA = {
    "version": "2.0",
    "metadata_inference": {"dataset_names": ["data"]},
    "dataset": {
        "data": {
            "sources": [{"type": "zarr", "dataset": "/absolute/path/to/data.zarr"}],
            "variables": ["2t"],
        },
    },
}

# The same named layout with a second dataset, to cover the multi-dataset iteration.
NAMED_MULTI_METADATA = {
    "version": "2.0",
    "metadata_inference": {"dataset_names": ["data", "ocean"]},
    "dataset": {
        "data": {"sources": [{"dataset": "/absolute/path/to/atmos.zarr"}], "variables": ["2t"]},
        "ocean": {"sources": [{"dataset": "/absolute/path/to/ocean.zarr"}], "variables": ["tos"]},
    },
}


def _run_sanitise(metadata, supporting_arrays=None):
    """Run SanitiseCmd.run against an in-memory metadata dict; returns what replace_metadata got."""
    if supporting_arrays is None:
        supporting_arrays = {}
    written = {}

    def fake_load_metadata(path, supporting_arrays=False):
        if supporting_arrays:
            return metadata, supporting_arrays
        return metadata

    def fake_replace_metadata(path, new_metadata, new_supporting_arrays=None, **kwargs):
        written["metadata"] = new_metadata
        written["supporting_arrays"] = new_supporting_arrays

    with (pytest.MonkeyPatch.context() as mp,):
        mp.setattr("anemoi.utils.checkpoints.load_metadata", fake_load_metadata)
        mp.setattr("anemoi.utils.checkpoints.replace_metadata", fake_replace_metadata)
        SanitiseCmd().run(Namespace(path="dummy.ckpt"))

    return written


def test_flat_metadata_is_sanitised():
    """The legacy flat layout keeps working: the patch branch runs and rewrites the file."""
    written = _run_sanitise(FLAT_METADATA)
    assert written, "flat metadata changed, so replace_metadata must have been called"
    # sanitise() shortens absolute paths; the rewritten metadata must not carry the full path
    assert "/absolute/path/to/data.zarr" not in str(written["metadata"])
    assert "data.zarr" in str(written["metadata"])


def test_named_metadata_is_sanitised():
    """The named-dataset layout (issue #618) no longer trips the `sources` assertion."""
    written = _run_sanitise(NAMED_METADATA)
    assert written, "named metadata changed, so replace_metadata must have been called"
    assert "/absolute/path/to/data.zarr" not in str(written["metadata"])
    assert "data.zarr" in str(written["metadata"])


def test_named_metadata_with_multiple_datasets_is_sanitised():
    """Every named dataset is checked for `sources` and rewritten, not just the first."""
    written = _run_sanitise(NAMED_MULTI_METADATA)
    assert written, "multi-dataset metadata changed, so replace_metadata must have been called"
    assert "/absolute/path/to/atmos.zarr" not in str(written["metadata"])
    assert "/absolute/path/to/ocean.zarr" not in str(written["metadata"])
    assert "atmos.zarr" in str(written["metadata"])
    assert "ocean.zarr" in str(written["metadata"])


def test_iter_dataset_metadatas_flat():
    assert list(_iter_dataset_metadatas(FLAT_METADATA)) == [("data", FLAT_METADATA["dataset"])]


def test_iter_dataset_metadatas_named():
    result = dict(_iter_dataset_metadatas(NAMED_METADATA))
    assert list(result) == ["data"]
    assert result["data"] is NAMED_METADATA["dataset"]["data"]


def test_iter_dataset_metadatas_named_multi():
    result = dict(_iter_dataset_metadatas(NAMED_MULTI_METADATA))
    assert set(result) == {"data", "ocean"}
    assert result["data"] is NAMED_MULTI_METADATA["dataset"]["data"]
    assert result["ocean"] is NAMED_MULTI_METADATA["dataset"]["ocean"]


def test_named_dataset_missing_sources_fails_with_named_error():
    """A named dataset without `sources` reports the dataset name, not a bare AssertionError."""
    # the absolute path (in a non-sources field) is what makes sanitise() change the metadata,
    # which is what routes the command into the patch branch that runs the `sources` check
    metadata = {
        "version": "2.0",
        "metadata_inference": {"dataset_names": ["data"]},
        "dataset": {"data": {"variables": ["2t"], "train_path": "/absolute/path/to/train.zarr"}},
    }
    with pytest.raises(AssertionError, match="dataset `data` is missing `sources`"):
        _run_sanitise(metadata)


def test_flat_metadata_missing_sources_fails():
    """The flat layout without `sources` still fails the check."""
    metadata = {
        "version": "2.0",
        "dataset": {"variables": ["2t"], "train_path": "/absolute/path/to/train.zarr"},
    }
    with pytest.raises(AssertionError, match="dataset `data` is missing `sources`"):
        _run_sanitise(metadata)


def test_already_sanitised_metadata_is_untouched():
    """When sanitise() leaves the metadata unchanged, replace_metadata is not called."""
    metadata = {
        "version": "2.0",
        "dataset": {
            "sources": [{"dataset": "already.relative.zarr"}],
        },
    }
    written = _run_sanitise(metadata)
    assert not written, "unchanged metadata must not be rewritten"
