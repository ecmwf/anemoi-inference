# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints

# from dummy import dummy_checkpoints


HERE = os.path.dirname(__file__)


@fake_checkpoints
def test_inference_dummy() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        os.path.join(HERE, "configs/simple.yaml"),
        overrides=dict(device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    runner.execute()


@fake_checkpoints
def test_inference_mars() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        os.path.join(HERE, "configs/simple.yaml"),
        overrides=dict(device="cpu", input="mars"),
    )
    runner = create_runner(config)
    runner.execute()


@fake_checkpoints
def test_inference_cds() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        os.path.join(HERE, "configs/simple.yaml"),
        overrides=dict(device="cpu", input={"cds": dict(dataset="reanalysis-era5-complete")}),
    )
    runner = create_runner(config)
    runner.execute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
