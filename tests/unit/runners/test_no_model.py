# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.runners.testing import NoModelRunner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import float_hash
from anemoi.inference.testing.mock_model import VariableOutputSetting


@pytest.mark.parametrize(
    ("output_mode", "expected"),
    [
        (VariableOutputSetting.ONES, lambda name, date: 1.0),
        (VariableOutputSetting.HASH, float_hash),
    ],
)
@fake_checkpoints
def test_no_model_runner_execute_outputs(
    output_mode: VariableOutputSetting, expected: Callable[[str, datetime.datetime], float]
) -> None:
    config = RunConfiguration(
        checkpoint="unit/checkpoints/simple.ckpt",
        date=-1,
        runner="no-model",
        device="cpu",
        input="dummy",
        lead_time="12h",
    )
    runner = create_runner(config, output_mode=output_mode)
    assert isinstance(runner, NoModelRunner)

    forecast = runner.forecast
    steps = 0

    def check_forecast(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
        nonlocal steps
        for states in forecast(*args, **kwargs):
            for state in states.values():
                assert state["fields"]
                for name, field in state["fields"].items():
                    assert (field == expected(name, state["date"])).all(), name
            steps += 1
            yield states

    runner.forecast = check_forecast

    with patch("torch.load", side_effect=AssertionError("NoModelRunner must not load the checkpoint's model")):
        runner.execute()

    assert steps == 2
