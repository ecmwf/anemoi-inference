# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

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
        metadata = checkpoint.multi_dataset_metadata

    assert any(
        "did not exist" in record.message and "dataset.variable_metadata" in record.message for record in caplog.records
    )

    # the new key is warned about but still applied (adding missing keys can be deliberate)
    patched = next(iter(metadata.values()))
    assert patched._metadata["dataset"]["variable_metadata"] == {"2t": "patched"}


@fake_checkpoints
def test_patch_metadata_no_warning_for_existing_keys(caplog) -> None:
    """A patch that only updates existing metadata keys must not emit the new-key warning."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint(
        "unit/checkpoints/simple.ckpt",
        patch_metadata={"dataset": {"shape": [1, 2, 3, 4]}},
    )
    with caplog.at_level(logging.WARNING):
        checkpoint.multi_dataset_metadata

    assert not any("did not exist" in record.message for record in caplog.records)


@fake_checkpoints
def test_checkpoint() -> None:
    """Simple test to check that the Checkpoint doesn't crash on important properties."""
    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint("unit/checkpoints/simple.ckpt")
    checkpoint._metadata.select_variables(
        include=["prognostic"],
        exclude=["forcing", "computed", "diagnostic"],
    )
    checkpoint.timestep
    checkpoint.multi_dataset
    checkpoint.multi_dataset_metadata
    checkpoint.precision
    checkpoint._metadata


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
