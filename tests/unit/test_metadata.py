# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.metadata import Metadata
from anemoi.inference.testing.mock_checkpoint import _load_mock_metadata_dict
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata


@pytest.mark.parametrize(
    "initial, patch, expected",
    [
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"something": {"else": "123"}}}},
            {"dataset": "abc", "something": {"else": "123"}},
        ),
        (
            {"config": {"dataloader": [{"dataset": "abc"}, {"dataset": "xyz"}]}},
            {"config": {"dataloader": {"cutout": [{"dataset": "123"}, {"dataset": "456"}]}}},
            {"cutout": [{"dataset": "123"}, {"dataset": "456"}]},
        ),
        (
            {"config": {"dataloader": "abc"}},
            {"config": {"dataloader": "xyz"}},
            "xyz",
        ),
    ],
)
def test_patch(initial, patch, expected):
    metadata = Metadata(initial)
    raw_before = metadata._metadata.to_dict()
    assert raw_before.get("config", {}).get("dataloader") == initial.get("config", {}).get("dataloader")

    metadata.patch(patch)
    raw_after = metadata._metadata.to_dict()
    assert raw_after["config"]["dataloader"] == expected


@pytest.mark.parametrize(
    "initial, patch, expected_new_keys",
    [
        # updating an existing leaf does not create any new key
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"dataset": "xyz"}}},
            [],
        ),
        # a new nested subtree is reported by its top-most new key only
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"something": {"else": "123"}}}},
            ["config.dataloader.something"],
        ),
        # a new leaf alongside an existing one is reported on its own
        (
            {"a": {"b": 1, "c": 2}},
            {"a": {"c": 3, "d": 4}},
            ["a.d"],
        ),
        # a brand-new top-level key is reported
        (
            {"a": 1},
            {"b": 2},
            ["b"],
        ),
    ],
)
def test_patch_returns_new_keys(initial, patch, expected_new_keys):
    metadata = Metadata(initial)
    new_keys = metadata.patch(patch)
    assert new_keys == expected_new_keys


def test_constant_fields_patch():
    model_metadata = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    metadata = Metadata(model_metadata)

    fields = ["z", "sdor", "slor", "lsm"]
    metadata.patch({"dataset": {"constant_fields": fields}})
    raw = metadata._metadata.to_dict()
    assert raw["dataset"]["constant_fields"] == fields

    # check that the rest of the metadata is still the same after patching.
    # Compare against the migrated form of model_metadata (migration may add
    # schema_version and other keys that are not in the original raw dict).
    raw["dataset"].pop("constant_fields")
    expected = Metadata(model_metadata)._metadata.to_dict()
    assert expected == raw


def test_metadata_factory():
    single_raw = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    multi_raw = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)

    # Both single- and multi-dataset checkpoints now produce plain Metadata instances.
    # Pass raw dicts so Metadata.__init__ handles migration via MetadataRegistry.load.
    single_m = Metadata(single_raw)
    multi_m = Metadata(multi_raw)

    assert isinstance(single_m, Metadata)
    assert isinstance(multi_m, Metadata)

    # multi_dataset property reflects the number of datasets
    assert not single_m.multi_dataset
    assert multi_m.multi_dataset


def test_multi_metadata():
    multi_raw = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)
    # Pass raw dict so Metadata.__init__ handles migration via MetadataRegistry.load.
    metadata = Metadata(multi_raw)

    assert metadata.dataset_names == ["data"]

    # check that metadata properties are accessible
    for prop in [
        "timestep",
        "variable_to_input_tensor_index",
        "variable_to_output_tensor_index",
        "input_tensor_index_to_variable",
        "output_tensor_index_to_variable",
    ]:
        assert getattr(metadata, prop) is not None

    assert metadata.variable_categories() is not None


def test_open_dataset_args_kwargs_removes_unsupported_keys():
    """Test that trajectory key is removed from kwargs but frequency and drop are kept."""
    # Load a real checkpoint and patch it with the trajectory key
    model_metadata = _load_mock_metadata_dict("unit/checkpoints/atmos.json")

    # Add dataloader config with trajectory key that should be removed
    model_metadata["config"]["dataloader"] = {
        "training": {
            "dataset": "test_dataset",
            "start": 2021,
            "end": 2021,
            "frequency": "6h",
            "drop": [],
            "trajectory": None,
        }
    }

    metadata = Metadata(model_metadata)

    # Get the args and kwargs
    args, kwargs = metadata.open_dataset_args_kwargs(use_original_paths=True, from_dataloader="training")

    # Verify that trajectory is removed
    assert "trajectory" not in kwargs

    # Verify that frequency and drop are kept (they are valid parameters)
    assert kwargs.get("frequency") == "6h"
    assert kwargs.get("drop") == []

    # Verify that other keys are still present
    assert kwargs.get("dataset") == "test_dataset"
    assert kwargs.get("start") == 2021
    assert kwargs.get("end") == 2021


def test_open_dataset_args_kwargs_removes_unsupported_keys_multi_dataset():
    """Test that trajectory key is removed from kwargs but frequency and drop are kept for multi-dataset."""
    # Load a multi-dataset checkpoint and patch it with the trajectory key
    model_metadata = _load_mock_metadata_dict("unit/checkpoints/multi-single.json")

    # Add dataloader config with trajectory key that should be removed for multi-dataset structure
    model_metadata["config"]["dataloader"] = {
        "training": {
            "datasets": {
                "data": {
                    "dataset": "test_dataset_multi",
                    "start": 2022,
                    "end": 2022,
                    "frequency": "12h",
                    "drop": ["var1"],
                    "trajectory": None,
                }
            }
        }
    }

    metadata = Metadata(model_metadata)

    # Get the args and kwargs
    args, kwargs = metadata.open_dataset_args_kwargs(use_original_paths=True, from_dataloader="training")

    # Verify that trajectory is removed
    assert "trajectory" not in kwargs

    # Verify that frequency and drop are kept (they are valid parameters)
    assert kwargs.get("frequency") == "12h"
    assert kwargs.get("drop") == ["var1"]

    # Verify that other keys are still present
    assert kwargs.get("dataset") == "test_dataset_multi"
    assert kwargs.get("start") == 2022
    assert kwargs.get("end") == 2022
