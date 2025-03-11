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

# Set the current working directory to the location of this script.
# So we access the configuration files relative to this script.
os.chdir(HERE)


@fake_checkpoints
def test_inference_simple() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        os.path.join(HERE, "configs/simple.yaml"),
        overrides=dict(runner="testing", device="cpu", input="dummy", trace_path="trace.log"),
    )
    runner = create_runner(config)
    runner.execute()


@fake_checkpoints
def test_inference_mwd() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        os.path.join(HERE, "configs/mwd.yaml"),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    runner.execute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_inference_simple()
    # exit()
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
