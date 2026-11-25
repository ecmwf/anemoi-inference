# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from datetime import timedelta

import numpy as np

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.testing import fake_checkpoints


@fake_checkpoints
def test_patch_metadata_warns_on_new_keys(caplog) -> None:
    """A patch that creates non-existing metadata keys must warn, name them, yet still apply them."""
    from anemoi.inference.checkpoint import Checkpoint

    # `variable_metadata` is a typo for the real `variables_metadata` key: this is the
    # foot-gun from issue #452 where a stale patch silently adds keys instead of updating.
    checkpoint = Checkpoint(
        "unit/checkpoints/simple.ckpt",
        patch_metadata={"dataset": {"variable_metadata": {"2t": "patched"}}},
    )
    with caplog.at_level(logging.WARNING):
        metadata = checkpoint.metadata

    assert any(
        "did not exist" in record.message and "dataset.variable_metadata" in record.message for record in caplog.records
    )

    # the new key is warned about but still applied (adding missing keys can be deliberate)
    assert metadata["dataset"]["variable_metadata"] == {"2t": "patched"}


@fake_checkpoints
def test_patch_metadata_no_warning_for_existing_keys(caplog) -> None:
    """A patch that only updates existing metadata keys must not emit the new-key warning."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint(
        "unit/checkpoints/simple.ckpt",
        patch_metadata={"dataset": {"shape": [1, 2, 3, 4]}},
    )
    with caplog.at_level(logging.WARNING):
        checkpoint.metadata

    assert not any("did not exist" in record.message for record in caplog.records)


@fake_checkpoints
def test_checkpoint() -> None:
    """Simple test to check that the Checkpoint doesn't crash on important properties."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint("unit/checkpoints/simple.ckpt")
    checkpoint.multi_dataset_metadata["data"].select_variables(
        include=["prognostic"],
        exclude=["forcing", "computed", "diagnostic"],
    )
    assert checkpoint.metadata.dataset_names == ["data"]
    assert checkpoint.timestep == timedelta(hours=6)
    assert not checkpoint.multi_dataset
    assert not checkpoint.richer_batch
    assert checkpoint.precision == "16-mixed"
    assert list(checkpoint.multi_dataset_metadata) == ["data"]


@fake_checkpoints
def test_replace_metadata_replaces_arrays() -> None:
    """New arrays are read through the checkpoint, the other arrays and datasets are kept."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint("unit/checkpoints/simple.ckpt")
    before = checkpoint.multi_dataset_metadata["data"]
    assert checkpoint.metadata.dataset("data").arrays.latitudes is None
    assert before.number_of_grid_points == 40320  # recorded in the metadata

    checkpoint.replace_metadata(
        supporting_arrays={"data": {"latitudes": np.arange(5.0), "longitudes": np.arange(5.0) * 2}},
    )

    after = checkpoint.multi_dataset_metadata["data"]
    assert after is not before
    assert list(checkpoint.metadata.dataset("data").arrays.latitudes) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(after.dataset_view.arrays.longitudes) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert after.number_of_grid_points == 5
    assert after.input_shape[2] == 5


def test_replace_metadata_rebuilds_the_wrapper_given_at_construction(multi_core) -> None:
    """A checkpoint built from a wrapper reads the new metadata after `replace_metadata`, not the old wrapper."""
    from anemoi.inference.metadata import InferenceMetadata

    checkpoint = Checkpoint(InferenceMetadata(multi_core))
    assert not checkpoint.richer_batch

    checkpoint.replace_metadata(patch={"version": "3.0"})

    assert checkpoint.richer_batch
    assert checkpoint.multi_dataset_metadata["data"].richer_batch
    assert checkpoint.metadata.version == "3.0"


def test_replace_metadata_updates_the_dataset_names(multi_core) -> None:
    """The dataset names are not cached from the old metadata."""
    checkpoint = Checkpoint(multi_core)
    assert checkpoint.dataset_names == ["data"]
    assert list(checkpoint.multi_dataset_metadata) == ["data"]

    raw = multi_core.to_dict()
    extra = {
        "metadata_inference": {"dataset_names": ["data", "extra"], "extra": raw["metadata_inference"]["data"]},
        "dataset": {"extra": raw["dataset"]["data"]},
    }
    checkpoint.replace_metadata(patch=extra)

    assert checkpoint.dataset_names == ["data", "extra"]
    assert list(checkpoint.multi_dataset_metadata) == ["data", "extra"]


def test_checkpoint_time_stepping(atmos_core) -> None:
    """The checkpoint's forwards read the task of the metadata."""
    checkpoint = Checkpoint(atmos_core)

    assert checkpoint.task == "forecaster"
    assert checkpoint.timestep == timedelta(hours=24)
    assert checkpoint.multi_step_input == 2
    assert checkpoint.multi_step_output == 1
    assert checkpoint.lagged == [timedelta(hours=-24), timedelta(0)]
    assert checkpoint.output_offsets == [timedelta(hours=24)]
    assert checkpoint.rollout_shift == timedelta(hours=24)


def test_checkpoint_explicit_times(atmos_core, multi_core) -> None:
    # single-dataset checkpoints record them in the training config
    config = {"config": {"training": {"explicit_times": {"input": [0, 6], "target": [1, 2, 3]}}}}
    checkpoint = Checkpoint(atmos_core.with_overrides(config))
    assert checkpoint.input_explicit_times == [0, 6]
    assert checkpoint.target_explicit_times == [1, 2, 3]

    # multi-dataset checkpoints record the relative date indices
    patch = {
        "metadata_inference": {
            "data": {"timesteps": {"input_relative_date_indices": [0, 2], "output_relative_date_indices": [1]}}
        }
    }
    checkpoint = Checkpoint(multi_core.with_overrides(patch))
    assert checkpoint.input_explicit_times == [0, 2]
    assert checkpoint.target_explicit_times == [1]


def test_checkpoint_temporal_downscaler_looks_ahead(multi_core) -> None:
    """For a temporal downscaler the task gives `lagged` and the offsets as training defines them."""
    patch = {
        "metadata_inference": {"task": "temporal-downscaler"},
        "config": {"task": {"input_timestep": "6h", "output_timestep": "2h", "output_right_boundary": True}},
    }
    checkpoint = Checkpoint(multi_core.with_overrides(patch))
    assert checkpoint.task == "temporal-downscaler"
    assert checkpoint.lagged == [timedelta(0), timedelta(hours=6)]
    assert checkpoint.output_offsets == [timedelta(hours=2), timedelta(hours=4), timedelta(hours=6)]
    assert checkpoint.rollout_shift == timedelta(hours=6)


@fake_checkpoints
def test_replace_metadata_leaves_no_stale_values() -> None:
    """Values cached from the old metadata (timestep, typed variables) are the new ones after the call."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint("unit/checkpoints/simple.ckpt")
    assert checkpoint.timestep == timedelta(hours=6)
    assert checkpoint.lagged == [timedelta(hours=-6), timedelta(0)]
    assert checkpoint.multi_dataset_metadata["data"].dataset_view.data.typed_variables()["2t"].param == "2t"

    checkpoint.replace_metadata(
        patch={
            "config": {"data": {"timestep": "3h"}},
            "dataset": {"variables_metadata": {"2t": {"mars": {"param": "167", "levtype": "sfc"}}}},
        }
    )

    assert checkpoint.timestep == timedelta(hours=3)
    assert checkpoint.lagged == [timedelta(hours=-3), timedelta(0)]
    assert checkpoint.multi_dataset_metadata["data"].dataset_view.data.typed_variables()["2t"].param == "167"
    assert checkpoint.metadata.task.timestep() == "3h"


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()


@fake_checkpoints
def test_update_metadata_from_zarr_replaces_the_dataset_metadata(mocker) -> None:
    """The metadata read from the training dataset is returned and becomes the metadata of the checkpoint."""
    import copy

    from anemoi.inference.checkpoint import Checkpoint
    from anemoi.inference.metadata import InferenceMetadata

    checkpoint = Checkpoint("unit/checkpoints/simple.ckpt")
    assert checkpoint.multi_dataset_metadata["data"].dataset_view.data.typed_variables()["2t"].param == "2t"

    dataset_metadata = copy.deepcopy(checkpoint.metadata.to_dict()["dataset"])
    dataset_metadata["sources"] = []
    dataset_metadata["variables_metadata"]["2t"] = {"mars": {"param": "167", "levtype": "sfc"}}
    dataset = mocker.MagicMock()
    dataset.metadata.return_value = dataset_metadata
    dataset.supporting_arrays.return_value = {"latitudes": np.arange(5.0), "longitudes": np.arange(5.0)}
    mocker.patch.object(InferenceMetadata, "open_dataset", return_value=dataset)

    new_metadata, new_arrays = checkpoint.update_metadata_from_zarr()

    assert new_metadata["dataset"] == dataset_metadata
    assert list(new_arrays["latitudes"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert checkpoint.multi_dataset_metadata["data"].dataset_view.data.typed_variables()["2t"].param == "167"
    assert checkpoint.multi_dataset_metadata["data"].number_of_grid_points == 5
