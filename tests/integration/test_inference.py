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
from omegaconf import OmegaConf

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

# each model can have more than one test configuration, defined as a listconfig in config.yaml
# the integration test is parameterised over the models and their test configurations
MODEL_CONFIGS = [
    (model, config) for model in MODELS for config in OmegaConf.load(INTEGRATION_ROOT / model / "config.yaml")
]


class Setup(NamedTuple):
    config: OmegaConf
    output: Path


@pytest.fixture(params=MODEL_CONFIGS, ids=[f"{model}/{config.name}" for model, config in MODEL_CONFIGS])
def test_setup(request, get_test_data: callable, tmp_path: Path) -> Setup:
    model, config = request.param
    input = config.input
    output = config.output
    inference_config = config.inference_config

    # download input file
    input_data = get_test_data(f"anemoi-integration-tests/inference/{model}/{input}")

    # prepare checkpoint
    metadata_path = INTEGRATION_ROOT / model / "metadata.json"
    checkpoint_path = tmp_path / Path("checkpoint.ckpt")
    save_fake_checkpoint(metadata_path, checkpoint_path)

    # to substitute inference config with real paths
    OmegaConf.register_new_resolver("input", lambda: str(input_data), replace=True)
    OmegaConf.register_new_resolver("output", lambda: str(tmp_path / output), replace=True)
    OmegaConf.register_new_resolver("checkpoint", lambda: str(checkpoint_path), replace=True)

    # save the inference config to disk
    inference_config = OmegaConf.to_yaml(inference_config, resolve=True)
    LOG.info(f"Resolved config:\n{inference_config}")

    with open(tmp_path / "integration_test.yaml", "w") as f:
        f.write(inference_config)

    return Setup(config=config, output=tmp_path / output)


def test_integration(test_setup: Setup, tmp_path) -> None:
    """Test the inference process using a fake checkpoint."""
    overrides = {"lead_time": "48h", "device": "cpu", "trace_path": "trace.log"}
    LOG.info(f"Config overrides: {overrides}")

    config = RunConfiguration.load(
        tmp_path / "integration_test.yaml",
        overrides=overrides,
    )
    runner = create_runner(config)
    runner.execute()

    assert (test_setup.output).exists(), "Output file was not created."

    # run the checks defined in the test configuration
    expected_params = runner._checkpoint.output_tensor_index_to_variable.values()

    for checks in test_setup.config.checks:
        for check, kwargs in checks.items():
            globals()[check](file=test_setup.output, expected_params=expected_params, **kwargs)


def check_grib(
    file: Path, expected_params: list, grib_keys: dict, check_accum: str = None, check_nans=False, **kwargs
) -> None:
    import earthkit.data as ekd
    import numpy as np

    ds = ekd.from_source("file", str(file))

    assert len(ds) > 0, "No fields found in the GRIB file."

    expected_params = [param.split("_")[0] for param in expected_params]

    params = set(ds.metadata("param"))
    assert (
        set(expected_params) == params
    ), f"Expected parameters {set(expected_params)} do not match found parameters {params}."

    for field in ds:
        for key, value in grib_keys.items():
            assert field[key] == value, f"Key {key} does not match expected value {value}."

    previous_field = None
    for field in ds.sel(param=expected_params[1]):
        assert np.all(field.values > 0), f"Field {field} is zero."

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
    if len(fields) < 2:
        raise ValueError(f"No fields found for accumulation check: {check_accum}")

    averages = [np.average(field.values) for field in fields]
    assert all(curr > prev for prev, curr in zip(averages, averages[1:])), f"{check_accum} is not accumulating"


def check_netcdf(file: Path, expected_params: list, check_accum: str = None, check_nans=False, **kwargs) -> None:
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(file)

    assert len(ds.data_vars) > 0, "No data found in the NetCDF file."

    skip = {"latitude", "longitude"}
    params = set(ds.data_vars) - skip
    assert (
        set(expected_params) == params
    ), f"Expected parameters {set(expected_params)} do not match found parameters {params}."

    for var in ds.data_vars:
        if var in skip:
            continue
        assert np.all(ds[var].values > 0), f"Variable {var} is zero."

        if check_nans:
            assert not np.isnan(ds[var].values).any(), f"Variable {var} contains NaN values."

    if not check_accum:
        return

    data_vars = list(ds.data_vars[check_accum])
    if len(data_vars) < 2:
        raise ValueError(f"No variables found for accumulation check: {check_accum}")

    averages = [np.average(data.values) for data in data_vars]
    assert all(curr > prev for prev, curr in zip(averages, averages[1:])), f"{check_accum} is not accumulating"
