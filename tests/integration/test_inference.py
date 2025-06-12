# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import NamedTuple

import pytest
from anemoi.utils.testing import get_test_data

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import save_fake_checkpoint

logging.basicConfig(level=logging.INFO)


def create_checkpoint(tmp_dir):
    """Fixture to create a fake checkpoint for testing."""
    repo_root = Path(__file__).resolve().parent
    metadata_path = repo_root / "checkpoints" / "single-o48-1.1.json"
    checkpoint_path = tmp_dir / Path("checkpoint.pth")

    save_fake_checkpoint(metadata_path, checkpoint_path)
    return checkpoint_path


def create_config_and_checkpoint(tmp_dir):
    """Fixture to create a fake configuration file and a checkpoint for testing."""

    checkpoint_path = create_checkpoint(tmp_dir)
    config_path = tmp_dir / "integration_test.yaml"

    with open(config_path, "w") as f:
        f.write(
            f"""
        lead_time: 48h
        post_processors:
            - accumulate_from_start_of_forecast
        checkpoint: {checkpoint_path}
        input:
            grib:  {tmp_dir}/input-single-o48-1.1.grib
        output:
            grib: {tmp_dir}/output.grib
        """
        )
    return config_path


class Setup(NamedTuple):
    config_path: Path
    tmp_dir: Path


@pytest.fixture
def test_setup() -> Setup:
    url_dataset = "anemoi-integration-tests/inference/input-single-o48-1.1.grib"
    grib_path = get_test_data(url_dataset)
    tmp_dir = Path(grib_path).parent
    config_path = create_config_and_checkpoint(tmp_dir)
    return Setup(config_path=config_path, tmp_dir=tmp_dir)


def test_inference_on_checkpoint(test_setup: Setup) -> None:
    """Test the inference process using a fake checkpoint."""
    config = RunConfiguration.load(
        test_setup.config_path,
        overrides=dict(device="cpu", trace_path="trace.log"),
    )
    runner = create_runner(config)
    runner.execute()

    assert (test_setup.tmp_dir / "output.grib").exists(), "Output GRIB file was not created."
