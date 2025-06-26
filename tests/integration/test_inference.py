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

    check_grib(
        test_setup.tmp_dir / "output.grib",
        grib_keys={
            "stream": "oper",
            "class": "ai",
            "type": "fc",
        },
        expected_params=[
            "lsm",
            "mcc",
            "hcc",
            "ssrd",
            "100v",
            "10v",
            "100u",
            "swvl2",
            "w",
            "strd",
            "sdor",
            "tp",
            "t",
            "2t",
            "stl2",
            "stl1",
            "sp",
            "sf",
            "10u",
            "v",
            "skt",
            "2d",
            "slor",
            "u",
            "z",
            "tcc",
            "ro",
            "cp",
            "tcw",
            "q",
            "msl",
            "lcc",
            "swvl1",
        ],
        check_accum="tp",
    )


def check_grib(file: Path, expected_params: list, grib_keys: dict, check_accum: str = None, check_nans=True) -> None:
    import earthkit.data as ekd
    import numpy as np

    ds = ekd.from_source("file", str(file))

    params = set(ds.metadata("param"))
    assert params == set(
        expected_params
    ), f"Expected parameters {expected_params} do not match found parameters {params}."

    for field in ds:
        for key, value in grib_keys.items():
            assert field[key] == value, f"Key {key} does not match expected value {value}."

    previous_field = None
    for field in ds.sel(param=expected_params[1]):
        if check_nans:
            assert not any(np.isnan(field.values)), f"Field {field} contains NaN values."

        if not previous_field:
            previous_field = field
            continue

        assert field["step"] > previous_field["step"] and field["valid_time"] > previous_field["valid_time"], (
            f"Field step {field['step']} (valid_time {field['valid_time']}) is not greater than previous field "
            f"step {previous_field['step']} (valid_time {previous_field['valid_time']})."
        )

    if not check_accum:
        return

    fields = list(ds.sel(param=check_accum))
    averages = [np.average(field.values) for field in fields]
    assert all(curr > prev for prev, curr in zip(averages, averages[1:])), f"{check_accum} is not accumulating"
