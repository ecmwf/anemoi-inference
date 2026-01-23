# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


@fake_checkpoints
def test_atmos() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        files_for_tests("unit/configs/atmos.yaml"),
        overrides=dict(device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    runner.execute()


@fake_checkpoints
def test_ocean() -> None:
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        files_for_tests("unit/configs/ocean.yaml"),
        overrides=dict(device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    runner.execute()


# @fake_checkpoints
# def test_coupled() -> None:
#     """Test the inference process using a fake checkpoint.

#     This function loads a configuration, creates a runner, and runs the inference
#     process to ensure that the system works as expected with the provided configuration.
#     """
#     config = CoupleConfiguration.load(files_for_tests("configs/coupled.yaml"))

#     global_config: Dict[str, Any] = {}

#     tasks = {name: create_task(name, action, global_config=global_config) for name, action in config.tasks.items()}

#     transport = create_transport(config.transport, config.couplings, tasks)

#     transport.start()
#     transport.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
