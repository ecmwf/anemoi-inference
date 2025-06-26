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

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import save_fake_checkpoint

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

INTEGRATION_ROOT = Path(__file__).resolve().parent

# each model has its own folder in the integration tests directory
# and contains at least a metadata.json and config.yaml file
MODELS = [
    path.name
    for path in INTEGRATION_ROOT.iterdir()
    if path.is_dir() and (path / "metadata.json").exists() and (path / "config.yaml").exists()
]


def create_config_and_checkpoint(tmp_dir: Path, model: str) -> Path:
    """Create a fake configuration file and a checkpoint for testing."""

    metadata_path = INTEGRATION_ROOT / model / "metadata.json"
    config_path = INTEGRATION_ROOT / model / "config.yaml"
    resolved_config_path = tmp_dir / "integration_test.yaml"

    checkpoint_path = tmp_dir / Path("checkpoint.ckpt")
    save_fake_checkpoint(metadata_path, checkpoint_path)

    with open(config_path, "r") as f:
        config = f.read()

    config = config.format(CHECKPOINT=checkpoint_path, TMP_DIR=tmp_dir)
    LOG.info(f"Resolved config:\n{config}\n")

    with open(resolved_config_path, "w") as f:
        f.write(config)

    return resolved_config_path


class Setup(NamedTuple):
    config_path: Path
    tmp_dir: Path


@pytest.fixture(params=MODELS)
def test_setup(request, get_test_data: callable) -> Setup:
    model = request.param
    grib_path = get_test_data(f"anemoi-integration-tests/inference/{model}/input.grib")
    tmp_dir = Path(grib_path).parent
    config_path = create_config_and_checkpoint(tmp_dir, model)
    return Setup(config_path=config_path, tmp_dir=tmp_dir)


def test_inference_on_checkpoint(test_setup: Setup) -> None:
    """Test the inference process using a fake checkpoint."""
    overrides = {"lead_time": "48h", "device": "cpu", "trace_path": "trace.log"}
    LOG.info(f"Config overrides: {overrides}")

    config = RunConfiguration.load(
        test_setup.config_path,
        overrides=overrides,
    )
    runner = create_runner(config)
    runner.execute()

    assert (test_setup.tmp_dir / "output.grib").exists(), "Output GRIB file was not created."
