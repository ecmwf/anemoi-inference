# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.metadata import Metadata
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.runners import create_runner, runner_registry
from anemoi.inference.runners.auto import AutoRunnerFactory
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.runners.temporal_downscaler import TemporalDownscalerMultiOutRunner
from anemoi.inference.testing import fake_checkpoints

LOG = logging.getLogger(__name__)

MULTI_DATASET_CHECKPOINT = "unit/checkpoints/multi-single.ckpt"
SINGLE_DATASET_CHECKPOINT = "unit/checkpoints/simple.ckpt"


class MarkerMetadata(Metadata):
    """Metadata subclass used to check that runner options reach the selected runner."""


@pytest.mark.parametrize(
    "patch, expected_class",
    [
        pytest.param({}, DefaultRunner, id="none"),
        pytest.param({"metadata_inference": {"task": None}}, DefaultRunner, id="unset-task"),
        pytest.param({"metadata_inference": {"task": "forecaster"}}, DefaultRunner, id="forecaster"),
        pytest.param({"metadata_inference": {"task": "default"}}, DefaultRunner, id="default"),
        pytest.param(
            {
                "metadata_inference": {
                    "task": "temporal_downscaler",
                    "data": {
                        "timesteps": {
                            "input_relative_date_indices": [0, 2],
                            "output_relative_date_indices": [1],
                        }
                    },
                }
            },
            TemporalDownscalerMultiOutRunner,
            id="temporal-downscaler",
        ),
    ],
)
@fake_checkpoints
def test_select_runner(patch: dict, expected_class: type) -> None:
    config = RunConfiguration(
        checkpoint=MULTI_DATASET_CHECKPOINT,
        date=-1,
        device="cpu",
        input="dummy",
        patch_metadata=patch,
    )
    assert isinstance(AutoRunnerFactory(config), expected_class)


@fake_checkpoints
def test_auto_is_default() -> None:
    config = RunConfiguration(checkpoint=SINGLE_DATASET_CHECKPOINT, device="cpu", input="dummy")

    assert runner_registry.lookup(config.runner) is AutoRunnerFactory

    runner = create_runner(config)
    assert isinstance(runner, DefaultRunner)


@fake_checkpoints
def test_auto_passes_on_runner_options() -> None:
    config = RunConfiguration(
        checkpoint=SINGLE_DATASET_CHECKPOINT,
        device="cpu",
        input="dummy",
        runner={"auto": {"classes": RunnerClasses(metadata=MarkerMetadata)}},
    )

    runner = create_runner(config)

    assert isinstance(runner, DefaultRunner)
    assert runner.classes.metadata is MarkerMetadata
