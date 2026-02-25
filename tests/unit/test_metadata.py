# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.metadata import MetadataFactory
from anemoi.inference.metadata import MultiDatasetMetadata
from anemoi.inference.metadata import SingleDatasetMetadata
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
    metadata = MetadataFactory(initial)
    assert metadata._metadata == initial

    metadata.patch(patch)
    assert metadata._metadata["config"]["dataloader"] == expected


def test_constant_fields_patch():
    model_metadata = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    metadata = MetadataFactory(model_metadata)

    fields = ["z", "sdor", "slor", "lsm"]
    metadata.patch({"dataset": {"constant_fields": fields}})
    assert metadata._metadata["dataset"]["constant_fields"] == fields

    # check that the rest of the metadata is still the same after patching
    metadata._metadata["dataset"].pop("constant_fields")
    assert model_metadata == metadata._metadata


def test_metadata_factory():
    single_metadata = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    multi_metadata = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)

    assert isinstance(MetadataFactory(single_metadata), SingleDatasetMetadata)
    assert isinstance(MetadataFactory(multi_metadata), MultiDatasetMetadata)


def test_multi_metadata():
    multi_metadata = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)
    metadata = MultiDatasetMetadata(multi_metadata)
    base_metadata = super(MultiDatasetMetadata, MultiDatasetMetadata(multi_metadata))

    assert metadata.dataset_names == ["data"]

    # check that multi-dataset metadata derived from the new `metadata_inference` matches the legacy metadata
    for property in [
        "timestep",
        "variable_to_input_tensor_index",
        "variable_to_output_tensor_index",
        "input_tensor_index_to_variable",
        "output_tensor_index_to_variable",
    ]:
        assert getattr(metadata, property) == getattr(base_metadata, property)

    for function in ["variable_categories"]:
        assert getattr(metadata, function)() == getattr(base_metadata, function)()


def test_open_dataset_args_kwargs_removes_unsupported_keys():
    """Test that trajectory key is removed from kwargs but frequency and drop are kept."""
    # Load a real checkpoint and patch it with the trajectory key
    model_metadata = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)

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

    metadata = MetadataFactory(model_metadata)

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
    model_metadata = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)

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

    metadata = MetadataFactory(model_metadata)

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
