# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.testing import TEST_DATA_URL
from anemoi.utils.testing import GetTestData
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import save_fake_checkpoint
from anemoi.inference.testing import testing_registry

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

INTEGRATION_ROOT = Path(__file__).resolve().parent


def _markers(config: DictConfig):
    """Add markers at collection time from `markers` entry in the config.
    Note that to avoid pytest showing a warning, markers must also be registered in the conftest.py
    """
    markers = config.get("markers", [])
    if not isinstance(markers, (list, ListConfig)):
        markers = [markers]
    return [getattr(pytest.mark, marker) for marker in markers]


# each model has its own folder in the integration tests directory
# and contains at least a metadata.json and config.yaml file
MODELS = [
    path.name
    for path in INTEGRATION_ROOT.iterdir()
    if path.is_dir() and (path / "metadata.json").exists() and (path / "config.yaml").exists()
]

# each model can have more than one test configuration, defined as a listconfig in config.yaml
# the integration test is parameterised over the models and their test configurations
# with optional markers (see _markers)
MODEL_CONFIGS = (
    pytest.param(
        (model, config),
        id=f"{model}/{config.name}",
        marks=_markers(config),
    )
    for model in MODELS
    for config in OmegaConf.load(INTEGRATION_ROOT / model / "config.yaml")
)


class Setup(NamedTuple):
    config: OmegaConf
    output: Path


@pytest.fixture(params=MODEL_CONFIGS)
def test_setup(request, get_test_data: GetTestData, tmp_path: Path) -> Setup:
    model, config = request.param
    input = config.input
    output = tmp_path / config.output
    inference_config = config.inference_config
    s3_path = f"anemoi-integration-tests/inference/{model}"

    # download input file(s)
    if not input:
        input = []
    if not isinstance(input, (list, ListConfig)):
        input = [input]
    input_data = [get_test_data(f"{s3_path}/{file}") for file in input]

    # change working directory to the input temporary directory, so the config could use relative paths to input files
    if input_data:
        workdir = Path(input_data[0]).parent
    else:
        workdir = tmp_path
    LOG.info(f"Changing working directory to {workdir}")
    os.chdir(workdir)

    # prepare checkpoint
    checkpoint_path = tmp_path / Path("checkpoint.ckpt")
    with open(INTEGRATION_ROOT / model / "metadata.json") as f:
        metadata = json.load(f)

    supporting_arrays = {}
    if array_names := list(metadata.get("supporting_arrays_paths", {})):

        def load_array(name):
            return name, np.load(get_test_data(f"{s3_path}/supporting-arrays/{name}.npy"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_array, array_names))

        supporting_arrays = dict(results)
        LOG.info(f"Supporting arrays: {supporting_arrays.keys()}")

    save_fake_checkpoint(metadata, checkpoint_path, supporting_arrays=supporting_arrays)

    # substitute inference config with real paths
    OmegaConf.register_new_resolver("input", lambda i=0: str(input_data[i]), replace=True)
    OmegaConf.register_new_resolver("output", lambda: str(output), replace=True)
    OmegaConf.register_new_resolver("checkpoint", lambda: str(checkpoint_path), replace=True)
    OmegaConf.register_new_resolver("s3", lambda: str(f"{TEST_DATA_URL}{s3_path}"), replace=True)
    OmegaConf.register_new_resolver("sys.prefix", lambda: sys.prefix, replace=True)

    # save the inference config to disk
    inference_config = OmegaConf.to_yaml(inference_config, resolve=True)
    LOG.info(f"Resolved config:\n{inference_config}")

    with open(tmp_path / "integration_test.yaml", "w") as f:
        f.write(inference_config)

    return Setup(config=config, output=output)


def test_integration(test_setup: Setup, tmp_path: Path) -> None:
    """Run the integration test suite."""

    overrides = {"lead_time": "48h", "device": "cpu"}
    LOG.info(f"Config overrides: {overrides}")

    config = RunConfiguration.load(
        tmp_path / "integration_test.yaml",
        overrides=overrides,
    )
    runner = create_runner(config)
    runner.execute()

    assert (test_setup.output).exists(), "Output file was not created."

    checkpoint_output_variables = _typed_variables_output(runner._checkpoint)
    LOG.info(f"Checkpoint output variables: {checkpoint_output_variables}")

    # run the checks defined in the test configuration
    # checks is a list of dicts, each with a single key-value pair
    for checks in test_setup.config.checks:
        check, kwargs = next(iter(checks.items()))
        # config can optionally pass expected output variables, by default it uses the checkpoint variables
        expected_variables_config = kwargs.pop("expected_variables", [])
        expected_variables = [
            VariableFromMarsVocabulary(var, {"param": var}) for var in expected_variables_config
        ] or checkpoint_output_variables

        testing_registry.create(
            check,
            file=test_setup.output,
            expected_variables=expected_variables,
            checkpoint=runner._checkpoint,
            **kwargs,
        )


def _typed_variables_output(checkpoint):
    output_variables = checkpoint.output_tensor_index_to_variable.values()
    return [checkpoint.typed_variables[name] for name in output_variables]
