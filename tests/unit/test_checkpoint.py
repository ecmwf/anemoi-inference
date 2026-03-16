# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.inference.testing import fake_checkpoints


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
