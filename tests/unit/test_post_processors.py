# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from typing import cast

import numpy as np
from pytest_mock import MockerFixture

from anemoi.inference.metadata import Metadata
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
    metadata.load_supporting_array.return_value = mask  # type: ignore
    processor = AssignMask(mocker.MagicMock(), metadata, mask="some_supporting_array")

    # check that load_supporting_array was called with the correct name
    metadata.load_supporting_array.assert_called_once_with("some_supporting_array")  # type: ignore

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
    metadata.load_supporting_array.assert_not_called()  # type: ignore

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
    metadata.load_supporting_array.return_value = mask  # type: ignore
    processor = ExtractMask(mocker.MagicMock(), metadata, mask="some_supporting_array")

    # check that load_supporting_array was called with the correct name
    metadata.load_supporting_array.assert_called_once_with("some_supporting_array")  # type: ignore

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
    metadata.load_supporting_array.assert_not_called()  # type: ignore

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
    metadata.load_supporting_array.assert_not_called()  # type: ignore

    # check that the indexer is set correctly
    np.testing.assert_equal(processor.indexer, extract_slice)

    # check that extraction works as expected
    new_state = processor.process(state)
    assert new_state["latitudes"].shape[0] == 25
    for field in new_state["fields"]:
        assert new_state["fields"][field].shape[0] == 25
        assert np.all(new_state["fields"][field] == state["fields"][field][extract_slice])
