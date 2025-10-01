# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import numpy as np
import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.inputs.cutout import _mask_and_combine_states
from anemoi.inference.runner import Runner
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


def test_mask_and_combine_states():
    states = [{"a": np.arange(5).astype(float)}, {"a": np.arange(5, 10).astype(float)}, {"a": np.arange(10, 15)}]
    masks = [np.zeros(5).astype(bool) for _ in states]
    masks[0][[1, 2]] = True
    masks[1][[2, 3]] = True
    masks[2][[2, 4]] = True

    combined_state: dict = {}
    for k in range(len(states)):
        mask = masks[k]
        new_state = states[k]
        combined_state = _mask_and_combine_states(combined_state, new_state, mask, ["a"])

    assert combined_state["a"].shape[0] == 6
    assert (
        combined_state["a"]
        == np.array(
            [
                states[0]["a"][1],
                states[0]["a"][2],
                states[1]["a"][2],
                states[1]["a"][3],
                states[2]["a"][2],
                states[2]["a"][4],
            ]
        )
    ).all()


@pytest.fixture
@fake_checkpoints
def runner() -> None:
    """Fake Runner for testing"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=dict(runner="testing", device="cpu", input="dummy", trace_path="trace.log"),
    )
    runner = create_runner(config)
    assert runner.checkpoint
    return runner


@fake_checkpoints
def test_cutout_no_mask(runner: Runner):
    from anemoi.inference.inputs.cutout import Cutout

    cutout_config = {
        "lam": {"mask": None, "dummy": {}},
        "global": {"mask": None, "dummy": {}},
    }
    cutout_input = Cutout(runner, variables=["2t"], **cutout_config)
    input_state = cutout_input.create_input_state(date=datetime.datetime.fromisoformat("2020-01-01T00:00"))
    number_of_grid_points = runner.checkpoint.number_of_grid_points

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == number_of_grid_points * 2

    assert all(input_state["_mask"]["lam"][slice(0, number_of_grid_points)])
    assert not all(input_state["_mask"]["lam"][slice(number_of_grid_points, None)])

    assert not all(input_state["_mask"]["global"][slice(0, number_of_grid_points)])
    assert all(input_state["_mask"]["global"][slice(number_of_grid_points, None)])


@fake_checkpoints
def test_cutout_with_slice(runner: Runner):
    from anemoi.inference.inputs.cutout import Cutout

    cutout_config = {
        "lam": {"mask": slice(0, 10), "dummy": {}},
        "global": {"mask": slice(10, 25), "dummy": {}},
    }
    cutout_input = Cutout(runner, variables=["2t"], **cutout_config)
    assert list(cutout_input.sources.keys()) == ["lam", "global"]

    input_state = cutout_input.create_input_state(date=datetime.datetime.fromisoformat("2020-01-01T00:00"))

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == 25

    assert all(input_state["_mask"]["lam"][slice(0, 10)])
    assert not all(input_state["_mask"]["lam"][slice(10, None)])

    assert not all(input_state["_mask"]["global"][slice(0, 10)])
    assert all(input_state["_mask"]["global"][slice(10, None)])


@fake_checkpoints
def test_cutout_with_array(runner: Runner):
    from anemoi.inference.inputs.cutout import Cutout

    number_of_grid_points = runner.checkpoint.number_of_grid_points

    lam_mask = np.zeros(number_of_grid_points, dtype=bool)
    lam_mask[:10] = True

    global_mask = np.zeros(number_of_grid_points, dtype=bool)
    global_mask[10:25] = True

    cutout_config = {
        "lam": {"mask": lam_mask, "dummy": {}},
        "global": {"mask": global_mask, "dummy": {}},
    }
    cutout_input = Cutout(runner, variables=["2t"], **cutout_config)
    input_state = cutout_input.create_input_state(date=datetime.datetime.fromisoformat("2020-01-01T00:00"))

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == 25

    assert all(input_state["_mask"]["lam"][slice(0, 10)])
    assert not all(input_state["_mask"]["lam"][slice(10, None)])

    assert not all(input_state["_mask"]["global"][slice(0, 10)])
    assert all(input_state["_mask"]["global"][slice(15, None)])
