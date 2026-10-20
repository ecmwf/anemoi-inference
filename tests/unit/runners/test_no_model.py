# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.runners.testing import NoModelRunner
from anemoi.inference.testing import fake_checkpoints


@fake_checkpoints
def test_no_model_runner_execute_outputs_ones() -> None:
    config = RunConfiguration(
        checkpoint="unit/checkpoints/simple.ckpt",
        date=-1,
        runner="no-model",
        device="cpu",
        input="dummy",
        lead_time="12h",
    )
    runner = create_runner(config)
    assert isinstance(runner, NoModelRunner)

    forecast = runner.forecast
    steps = 0

    def check_forecast(*args: Any, **kwargs: Any) -> Generator[dict, None, None]:
        nonlocal steps
        for states in forecast(*args, **kwargs):
            for state in states.values():
                assert state["fields"]
                for name, field in state["fields"].items():
                    assert (field == 1).all(), name
            steps += 1
            yield states

    runner.forecast = check_forecast

    with patch("torch.load", side_effect=AssertionError("NoModelRunner must not load the checkpoint's model")):
        runner.execute()

    assert steps == 2
