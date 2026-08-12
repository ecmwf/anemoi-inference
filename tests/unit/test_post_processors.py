# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from datetime import timedelta
from typing import cast

import numpy as np
import pytest
from pytest_mock import MockerFixture

from anemoi.inference.metadata import Metadata
from anemoi.inference.post_processors.accumulate import Accumulate
from anemoi.inference.post_processors.assign import AssignMask
from anemoi.inference.post_processors.extract import ExtractMask
from anemoi.inference.post_processors.extract import ExtractSlice
from anemoi.inference.types import State


def test_assign_mask_supporting_array(
    mocker: MockerFixture,
    state: State,
    assign_mask_npy: str,
):
    # mock the context to return the mask when load_supporting_array is called
    mask = np.load(assign_mask_npy)
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.load_supporting_array.return_value = mask
    processor = AssignMask(mocker.MagicMock(), metadata, mask="some_supporting_array")

    # check that load_supporting_array was called with the correct name
    metadata.load_supporting_array.assert_called_once_with("some_supporting_array")

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, mask)

    # check that assignment works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == mask.shape[0]
    assert np.isnan(new_state["latitudes"]).sum() == (~mask).sum()
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == mask.shape[0]
        assert np.isnan(new_state["fields"][field]).sum() == (~mask).sum()


def test_assign_mask_npy(
    mocker: MockerFixture,
    state: State,
    assign_mask_npy: str,
):
    mask = np.load(assign_mask_npy)

    # mock the context just because AssignMask requires it
    metadata = cast(Metadata, mocker.MagicMock())
    processor = AssignMask(mocker.MagicMock(), metadata, mask=assign_mask_npy)

    # check that nothing was done with the context
    metadata.load_supporting_array.assert_not_called()

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, mask)

    # check that assignment works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == mask.shape[0]
    assert np.isnan(new_state["latitudes"]).sum() == (~mask).sum()
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == mask.shape[0]
        assert np.isnan(new_state["fields"][field]).sum() == (~mask).sum()


def test_extract_mask_supporting_array(
    mocker: MockerFixture,
    state: State,
    extract_mask_npy: str,
):

    # mock the context to return the mask when load_supporting_array is called
    mask = np.load(extract_mask_npy)
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.load_supporting_array.return_value = mask
    processor = ExtractMask(mocker.MagicMock(), metadata, mask="some_supporting_array")

    # check that load_supporting_array was called with the correct name
    metadata.load_supporting_array.assert_called_once_with("some_supporting_array")

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, mask)

    # check that extraction works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == mask.sum()
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == mask.sum()
        assert np.all(new_state["fields"][field] == state["fields"][field][mask])


def test_extract_mask_npy(
    mocker: MockerFixture,
    state: State,
    extract_mask_npy: str,
):
    mask = np.load(extract_mask_npy)

    # mock the context just because ExtractMask requires it
    metadata = cast(Metadata, mocker.MagicMock())
    processor = ExtractMask(mocker.MagicMock(), metadata, mask=extract_mask_npy)

    # check that nothing was done with the context
    metadata.load_supporting_array.assert_not_called()

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, mask)

    # check that extraction works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == mask.sum()
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == mask.sum()
        assert np.all(new_state["fields"][field] == state["fields"][field][mask])


def test_extract_slice(
    mocker: MockerFixture,
    state: State,
):
    slice_args = (0, 25)
    extract_slice = slice(*slice_args)

    # mock the context just because ExtractSlice requires it
    metadata = cast(Metadata, mocker.MagicMock())
    processor = ExtractSlice(mocker.MagicMock(), metadata, *slice_args)

    # check that nothing was done with the context
    metadata.load_supporting_array.assert_not_called()

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, extract_slice)

    # check that extraction works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == 25
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == 25
        assert np.all(new_state["fields"][field] == state["fields"][field][extract_slice])


@pytest.fixture
def initial_state(state):
    """State representing the initial condition (step=0), without accumulation fields."""
    s = state.copy()
    s["step"] = timedelta(0)
    return s


def _make_accumulate(mocker, accumulations=("tp",), allow_negative=False, emit_initial_zeros=False, n_points=50):
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.accumulations = list(accumulations)
    metadata.number_of_grid_points = n_points
    return Accumulate(
        mocker.MagicMock(),
        metadata,
        accumulations=list(accumulations),
        allow_negative=allow_negative,
        emit_initial_zeros=emit_initial_zeros,
    )


def test_accumulate_default_no_zeros_at_step_zero(mocker: MockerFixture, initial_state: State):
    """By default (emit_initial_zeros=False), no zeros are emitted at step=0."""
    processor = _make_accumulate(mocker)

    new_state = processor.process(initial_state)

    assert "tp" not in new_state["fields"]


def test_accumulate_emit_initial_zeros_missing_field(mocker: MockerFixture, initial_state: State):
    """With emit_initial_zeros=True, zero-valued fields are emitted at step=0 even when absent from the state."""
    processor = _make_accumulate(mocker, emit_initial_zeros=True)

    assert "tp" not in initial_state["fields"]
    new_state = processor.process(initial_state)

    assert "tp" in new_state["fields"]
    np.testing.assert_array_equal(new_state["fields"]["tp"], 0.0)
    assert new_state["start_steps"]["tp"] == timedelta(0)
    # non-accumulation fields are unchanged
    np.testing.assert_array_equal(new_state["fields"]["2t"], initial_state["fields"]["2t"])


def test_accumulate_emit_initial_zeros_existing_field(mocker: MockerFixture, initial_state: State):
    """With emit_initial_zeros=True, accumulation fields already in the state are overridden to zero at step=0."""
    initial_state["fields"]["tp"] = np.ones(len(initial_state["latitudes"]))
    processor = _make_accumulate(mocker, emit_initial_zeros=True)

    new_state = processor.process(initial_state)

    np.testing.assert_array_equal(new_state["fields"]["tp"], 0.0)
    assert new_state["start_steps"]["tp"] == timedelta(0)


def test_accumulate_emit_initial_zeros_skipped_when_not_step_zero(mocker: MockerFixture, state: State):
    """With emit_initial_zeros=True, zeros are not emitted when the first call is not at step=0."""
    n = len(state["latitudes"])
    state["fields"]["tp"] = np.full(n, 3.0)
    # state fixture has step=timedelta(hours=6), not step=0
    processor = _make_accumulate(mocker, emit_initial_zeros=True, n_points=n)

    new_state = processor.process(state)

    # zeros not emitted; accumulation started from this step
    np.testing.assert_array_almost_equal(new_state["fields"]["tp"], 3.0)
