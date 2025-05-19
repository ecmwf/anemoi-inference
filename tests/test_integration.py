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

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import save_fake_checkpoint


@pytest.fixture(autouse=True)
def set_working_directory() -> None:
    """Automatically set the working directory to the repo root."""
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    os.chdir(repo_root)


# @pytest.fixture()
def checkpoint_path(tmp_path):
    """Fixture to create a fake checkpoint for testing."""
    path = tmp_path / "checkpoint.pth"
    metadata_path = Path("tests/checkpoints/atmos.json")
    save_fake_checkpoint(metadata_path, path)
    return path


# @pytest.fixture
def config_path(tmp_path):
    """Fixture to create a fake configuration file for testing."""
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        f.write(
            """
            checkpoint: checkpoint.pth
            input:
                grib: input.grib
            output:
                grib: output.grib
            """
        )
    return path


def test_inference_on_checkpoint(config_path):
    """Test the inference process using a fake checkpoint.

    This function loads a configuration, creates a runner, and runs the inference
    process to ensure that the system works as expected with the provided configuration.
    """
    config = RunConfiguration.load(
        config_path,
        overrides=dict(runner="testing", device="cpu", input="dummy", trace_path="trace.log"),
    )
    runner = create_runner(config)
    runner.execute()


if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    config_path(Path(""))
    checkpoint_path(Path(""))

    config = RunConfiguration.load(
        "integration_test.yaml",
        overrides=dict(runner="testing", device="cpu", input="dummy", trace_path="trace.log"),
    )
    runner = create_runner(config)
    runner.execute()
