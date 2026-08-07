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


def _make_accumulate(mocker, accumulations=("tp",), allow_negative=False):
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.accumulations = list(accumulations)
    return Accumulate(mocker.MagicMock(), metadata, accumulations=list(accumulations), allow_negative=allow_negative)


def test_accumulate_step_zero_missing_field(mocker: MockerFixture, initial_state: State):
    """At step=0, zero-valued fields are emitted for accumulation variables not present in the state."""
    processor = _make_accumulate(mocker)

    assert "tp" not in initial_state["fields"]
    new_state = processor.process(initial_state)

    assert "tp" in new_state["fields"]
    np.testing.assert_array_equal(new_state["fields"]["tp"], 0.0)
    assert new_state["start_steps"]["tp"] == timedelta(0)
    # non-accumulation fields are unchanged
    np.testing.assert_array_equal(new_state["fields"]["2t"], initial_state["fields"]["2t"])


def test_accumulate_step_zero_existing_field(mocker: MockerFixture, initial_state: State):
    """At step=0, accumulation fields already in the state are overridden to zero."""
    initial_state["fields"]["tp"] = np.ones(len(initial_state["latitudes"]))
    processor = _make_accumulate(mocker)

    new_state = processor.process(initial_state)

    np.testing.assert_array_equal(new_state["fields"]["tp"], 0.0)
    assert new_state["start_steps"]["tp"] == timedelta(0)
