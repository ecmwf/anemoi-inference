# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from anemoi.metadata import DatasetArrays

from anemoi.inference.inputs.dataset import DatasetInput

_CTX = SimpleNamespace(verbosity=0, reference_date=None)
_META = SimpleNamespace(dataset_name="test_dataset", dataset_view=SimpleNamespace(arrays=DatasetArrays()))
_KWARGS = dict(variables=[], open_dataset_args=(), open_dataset_kwargs={})


def test_warning_logged_when_pre_processors_configured():
    with pytest.warns(UserWarning, match="will NOT be applied"):
        DatasetInput(_CTX, _META, pre_processors=[{"scale": {}}], **_KWARGS)


def test_no_warning_when_no_pre_processors(recwarn):
    DatasetInput(_CTX, _META, **_KWARGS)
    assert not any("will NOT be applied" in str(w.message) for w in recwarn.list)


def test_uses_warn_once_when_available():
    call_count = [0]

    def mock_warn_once(msg):
        call_count[0] += 1
        warnings.warn(msg, UserWarning)

    ctx = SimpleNamespace(verbosity=0, reference_date=None, _warn_once=mock_warn_once)
    with pytest.warns(UserWarning, match="will NOT be applied"):
        DatasetInput(ctx, _META, pre_processors=[{"scale": {}}], **_KWARGS)
    assert call_count[0] == 1


def test_grid_indices_are_read_from_the_supporting_arrays():
    grid_indices = np.array([3, 1, 2])
    meta = SimpleNamespace(
        dataset_name="test_dataset",
        dataset_view=SimpleNamespace(arrays=DatasetArrays({"grid_indices": grid_indices})),
    )
    inp = DatasetInput(_CTX, meta, **_KWARGS)
    np.testing.assert_array_equal(inp.grid_indices, grid_indices)


def test_explicit_grid_indices_win_over_the_supporting_arrays():
    meta = SimpleNamespace(
        dataset_name="test_dataset",
        dataset_view=SimpleNamespace(arrays=DatasetArrays({"grid_indices": np.array([3, 1, 2])})),
    )
    inp = DatasetInput(_CTX, meta, grid_indices=[0, 1], **_KWARGS)
    assert list(inp.grid_indices) == [0, 1]


def test_no_grid_indices_selects_everything():
    inp = DatasetInput(_CTX, _META, **_KWARGS)
    assert inp.grid_indices == slice(None)
