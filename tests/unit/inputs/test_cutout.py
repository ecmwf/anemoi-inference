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
from anemoi.inference.metadata import InferenceMetadata
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
    """Runner for testing"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=dict(runner="default", device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    assert runner.checkpoint
    # since this is a single-dataset checkpoint, we can use the metadata of the first dataset for testing
    return runner, runner.checkpoint.multi_dataset_metadata[runner.checkpoint.dataset_names[0]]


@fake_checkpoints
def test_cutout_no_mask(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    cutout_config = [
        {"lam": {"mask": None, "dummy": {}}},
        {"global": {"mask": None, "dummy": {}}},
    ]
    cutout_input = Cutout(runner, metadata, variables=["2t"], sources=cutout_config)
    number_of_grid_points = metadata.number_of_grid_points
    input_state = cutout_input.create_input_state(dates=[datetime.datetime.fromisoformat("2020-01-01T00:00")])

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == number_of_grid_points * 2

    assert all(input_state["_mask"]["lam"][slice(0, number_of_grid_points)])
    assert not all(input_state["_mask"]["lam"][slice(number_of_grid_points, None)])

    assert not all(input_state["_mask"]["global"][slice(0, number_of_grid_points)])
    assert all(input_state["_mask"]["global"][slice(number_of_grid_points, None)])


@fake_checkpoints
def test_cutout_with_slice(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    cutout_config = [
        {"lam": {"mask": slice(0, 10), "dummy": {}}},
        {"global": {"mask": slice(10, 25), "dummy": {}}},
    ]
    cutout_input = Cutout(runner, metadata, variables=["2t"], sources=cutout_config)
    assert list(cutout_input.sources.keys()) == ["lam", "global"]
    input_state = cutout_input.create_input_state(dates=[datetime.datetime.fromisoformat("2020-01-01T00:00")])

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == 25

    assert all(input_state["_mask"]["lam"][slice(0, 10)])
    assert not all(input_state["_mask"]["lam"][slice(10, None)])

    assert not all(input_state["_mask"]["global"][slice(0, 10)])
    assert all(input_state["_mask"]["global"][slice(10, None)])


@fake_checkpoints
def test_cutout_with_array(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    number_of_grid_points = metadata.number_of_grid_points

    lam_mask = np.zeros(number_of_grid_points, dtype=bool)
    lam_mask[:10] = True

    global_mask = np.zeros(number_of_grid_points, dtype=bool)
    global_mask[10:25] = True

    cutout_config = [{"lam": {"mask": lam_mask, "dummy": {}}}, {"global": {"mask": global_mask, "dummy": {}}}]
    cutout_input = Cutout(runner, metadata, variables=["2t"], sources=cutout_config)
    number_of_grid_points = metadata.number_of_grid_points
    input_state = cutout_input.create_input_state(dates=[datetime.datetime.fromisoformat("2020-01-01T00:00")])

    assert "_mask" in input_state
    assert input_state["latitudes"].shape[0] == 25

    assert all(input_state["_mask"]["lam"][slice(0, 10)])
    assert not all(input_state["_mask"]["lam"][slice(10, None)])

    assert not all(input_state["_mask"]["global"][slice(0, 10)])
    assert all(input_state["_mask"]["global"][slice(15, None)])


@fake_checkpoints
def test_cutout_masks_from_supporting_arrays(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    name = metadata.dataset_name
    number_of_grid_points = metadata.number_of_grid_points

    lam_mask = np.zeros(number_of_grid_points, dtype=bool)
    lam_mask[:10] = True
    global_mask = np.zeros(number_of_grid_points, dtype=bool)
    global_mask[10:25] = True

    runner.checkpoint.replace_metadata(
        supporting_arrays={name: {"lam/cutout_mask": lam_mask, "global/cutout_mask": global_mask}}
    )
    metadata = runner.checkpoint.multi_dataset_metadata[name]

    cutout_config = [{"lam": {"dummy": {}}}, {"global": {"dummy": {}}}]
    cutout_input = Cutout(runner, metadata, variables=["2t"], sources=cutout_config)
    np.testing.assert_array_equal(cutout_input.masks["lam"], lam_mask)
    np.testing.assert_array_equal(cutout_input.masks["global"], global_mask)


@fake_checkpoints
def test_cutout_missing_mask_array_raises(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    with pytest.raises(ValueError, match="lam/cutout_mask"):
        Cutout(runner, metadata, variables=["2t"], sources=[{"lam": {"dummy": {}}}])


@fake_checkpoints
def test_cutout_coordinates_fall_back_to_the_source_supporting_arrays(runner: tuple[Runner, InferenceMetadata]):
    from anemoi.inference.inputs.cutout import Cutout

    runner, metadata = runner
    name = metadata.dataset_name
    n = metadata.number_of_grid_points

    lam_mask = np.zeros(n, dtype=bool)
    lam_mask[:10] = True
    global_mask = np.zeros(n, dtype=bool)
    global_mask[10:25] = True
    lats = {i: np.arange(n, dtype=float) + 1000 * i for i in range(2)}
    lons = {i: -np.arange(n, dtype=float) - 1000 * i for i in range(2)}

    runner.checkpoint.replace_metadata(
        supporting_arrays={
            name: {
                f"source{i}/{key}": values[i]
                for i in range(2)
                for key, values in (("latitudes", lats), ("longitudes", lons))
            }
        }
    )
    metadata = runner.checkpoint.multi_dataset_metadata[name]

    cutout_input = Cutout(
        runner,
        metadata,
        variables=["2t"],
        sources=[{"lam": {"mask": lam_mask, "dummy": {}}}, {"global": {"mask": global_mask, "dummy": {}}}],
    )

    # the sources give no coordinates: they are read from the `source<i>/latitudes|longitudes` arrays
    for source in cutout_input.sources.values():
        create_input_state = source.create_input_state

        def without_coordinates(*, dates, _create=create_input_state, **kwargs):
            state = _create(dates=dates, **kwargs)
            return {k: v for k, v in state.items() if k != "latitudes"}

        source.create_input_state = without_coordinates

    state = cutout_input.create_input_state(dates=[datetime.datetime.fromisoformat("2020-01-01T00:00")])
    np.testing.assert_array_equal(state["latitudes"], np.concatenate([lats[0][lam_mask], lats[1][global_mask]]))
