# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the dates logic in DatasetInput.

Tests exercise `create_input_state` and `load_forcings_state` -- the
public API -- to verify that the correct date slices are loaded from
the underlying dataset.
"""

from types import SimpleNamespace
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pytest

from anemoi.inference.inputs.dataset import DatasetInput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARS = ["t", "u", "v"]
_DATES = np.array(
    [
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-01T06:00"),
        np.datetime64("2024-01-01T12:00"),
        np.datetime64("2024-01-01T18:00"),
        np.datetime64("2024-01-02T00:00"),
    ]
)
N_DATES = len(_DATES)
N_VARS = len(VARS)
N_ENS = 1
N_CELLS = 10
N_STEPS = 4

# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------


class _MockDataset:
    """Minimal mock that behaves like an anemoi dataset."""

    def __init__(self, data, *, has_base_dates=False):
        self._data = data
        self.shape = data.shape
        self.dates = _DATES
        self.frequency = np.timedelta64(6, "h")
        self.latitudes = np.linspace(-90, 90, N_CELLS)
        self.longitudes = np.linspace(0, 360, N_CELLS)
        self.variables = list(VARS)
        self.typed_variables = {v: SimpleNamespace(is_constant=False) for v in VARS}
        self.name_to_index = {v: i for i, v in enumerate(VARS)}

        if has_base_dates:
            self.base_dates = _DATES
            self.step_frequency = np.timedelta64(6, "h")

    def __getitem__(self, key):
        return self._data[key]


def _make_4d_data():
    return np.arange(N_DATES * N_VARS * N_ENS * N_CELLS, dtype=float).reshape(N_DATES, N_VARS, N_ENS, N_CELLS)


def _make_5d_data():
    return np.arange(N_DATES * N_VARS * N_ENS * N_STEPS * N_CELLS, dtype=float).reshape(
        N_DATES, N_VARS, N_ENS, N_STEPS, N_CELLS
    )


# ---------------------------------------------------------------------------
# Mock context / metadata
# ---------------------------------------------------------------------------

_TRACE = SimpleNamespace(from_input=lambda *a, **k: None)
_HANDLER = SimpleNamespace(trace=_TRACE)


def _make_ctx():
    return SimpleNamespace(
        verbosity=0,
        reference_date=None,
        tensor_handlers={"test_dataset": _HANDLER},
    )


def _make_meta(*, lagged=None):
    """Build mock metadata.

    lagged: list of timedelta offsets for lagged inputs (e.g. [0, -6h]).
            Defaults to a single lag of 0.
    """
    if lagged is None:
        lagged = [np.timedelta64(0, "h")]
    return SimpleNamespace(
        dataset_name="test_dataset",
        _supporting_arrays={},
        lagged=lagged,
        variable_to_input_tensor_index={v: i for i, v in enumerate(VARS)},
    )


# ---------------------------------------------------------------------------
# Helper to build a DatasetInput with the mock dataset injected
# ---------------------------------------------------------------------------


def _make_input(mock_ds, **extra_kwargs):
    """Construct a DatasetInput with *ds* patched to *mock_ds*."""
    ctx = _make_ctx()
    meta = _make_meta(**extra_kwargs.pop("meta_kwargs", {}))
    kwargs = dict(variables=list(VARS), open_dataset_args=(), open_dataset_kwargs={})
    kwargs.update(extra_kwargs)

    with patch.object(DatasetInput, "ds", new_callable=PropertyMock, return_value=mock_ds):
        inp = DatasetInput(ctx, meta, **kwargs)
    # Inject the mock into the instance dict so the cached_property descriptor
    # is bypassed on subsequent accesses without polluting the class.
    inp.__dict__["ds"] = mock_ds
    return inp


# ===================================================================
# 4D (analysis) tests -- these should pass on both old and new code
# ===================================================================


class TestCreateInputState4D:
    """Test create_input_state with a standard 4D dataset."""

    def test_returns_correct_fields_for_date(self):
        """Fields should contain data from the requested date."""
        data = _make_4d_data()
        ds = _MockDataset(data)
        inp = _make_input(ds)

        date = _DATES[2]
        state = inp.create_input_state(date=date)

        for vi, var in enumerate(VARS):
            # shape: (n_lagged, n_cells) -- single lag of 0, no ensemble dim
            expected = data[2, vi, 0, :]
            np.testing.assert_array_equal(state["fields"][var], expected.reshape(1, N_CELLS))

    def test_lagged_dates(self):
        """With lagged offsets the correct date slices should be loaded."""
        data = _make_4d_data()
        ds = _MockDataset(data)
        # lag -6h then 0  =>  date-6h then date (ascending order)
        lags = [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
        inp = _make_input(ds, meta_kwargs=dict(lagged=lags))

        date = _DATES[2]  # 2024-01-01T12:00
        state = inp.create_input_state(date=date)

        for vi, var in enumerate(VARS):
            # Two lagged dates: index 1 (T06) and index 2 (T12)
            expected = np.stack([data[1, vi, 0, :], data[2, vi, 0, :]], axis=0)
            np.testing.assert_array_equal(state["fields"][var], expected)

    def test_constant_loads_single_date(self):
        """constant=True should load only the requested date (no lags)."""
        data = _make_4d_data()
        ds = _MockDataset(data)
        lags = [np.timedelta64(0, "h"), np.timedelta64(-6, "h")]
        inp = _make_input(ds, meta_kwargs=dict(lagged=lags))

        date = _DATES[3]
        state = inp.create_input_state(date=date, constant=True)

        for vi, var in enumerate(VARS):
            expected = data[3, vi, 0, :].reshape(1, N_CELLS)
            np.testing.assert_array_equal(state["fields"][var], expected)

    def test_date_not_in_dataset_raises(self):
        """Requesting a date outside the dataset should raise ValueError."""
        ds = _MockDataset(_make_4d_data())
        inp = _make_input(ds)
        with pytest.raises(ValueError, match="not found"):
            inp.create_input_state(date=np.datetime64("2099-01-01"))


class TestLoadForcingsState4D:
    """Test load_forcings_state with a standard 4D dataset."""

    def test_single_date(self):
        data = _make_4d_data()
        ds = _MockDataset(data)
        inp = _make_input(ds)

        dates = [_DATES[1]]
        current_state = {"date": _DATES[1], "step": np.timedelta64(0, "h")}
        state = inp.load_forcings_state(dates=dates, current_state=current_state)

        # Result shape per variable: (n_dates, n_cells)
        for vi, var in enumerate(VARS):
            expected = data[1, vi, 0, :].reshape(1, N_CELLS)
            np.testing.assert_array_equal(state["fields"][var], expected)

    def test_multiple_consecutive_dates(self):
        data = _make_4d_data()
        ds = _MockDataset(data)
        inp = _make_input(ds)

        dates = [_DATES[1], _DATES[2], _DATES[3]]
        current_state = {"date": _DATES[1], "step": np.timedelta64(0, "h")}
        state = inp.load_forcings_state(dates=dates, current_state=current_state)

        for vi, var in enumerate(VARS):
            expected = data[1:4, vi, 0, :]
            np.testing.assert_array_equal(state["fields"][var], expected)


# ===================================================================
# 5D (trajectory) tests
# ===================================================================


class TestCreateInputState5D:
    """Test create_input_state with a 5D trajectory dataset."""

    def test_returns_step0_for_base_date(self):
        """At the base date itself, step 0 data should be returned."""
        data = _make_5d_data()
        ds = _MockDataset(data, has_base_dates=True)
        inp = _make_input(ds, use_trajectories=True)

        date = _DATES[0]
        state = inp.create_input_state(date=date)

        for vi, var in enumerate(VARS):
            expected = data[0, vi, 0, 0, :].reshape(1, N_CELLS)
            np.testing.assert_array_equal(state["fields"][var], expected)


class TestLoadForcingsState5D:
    """Test load_forcings_state with a 5D trajectory dataset."""

    def test_dates_after_base_use_trajectory(self):
        """Dates beyond the base date should be read from the trajectory dimension."""
        data = _make_5d_data()
        ds = _MockDataset(data, has_base_dates=True)
        inp = _make_input(ds, use_trajectories=True)

        base_date = _DATES[0]
        # Request 6h and 12h after base -> steps 1, 2
        dates = [
            base_date + np.timedelta64(6, "h"),
            base_date + np.timedelta64(12, "h"),
        ]
        current_state = {"date": base_date, "step": np.timedelta64(0, "h")}
        state = inp.load_forcings_state(dates=dates, current_state=current_state)

        for vi, var in enumerate(VARS):
            expected = data[0, vi, 0, 1:3, :]  # steps 1 and 2
            np.testing.assert_array_equal(state["fields"][var], expected)

    def test_dates_before_base_use_step0(self):
        """Dates <= base should each be loaded at step 0 of their own base date."""
        data = _make_5d_data()
        ds = _MockDataset(data, has_base_dates=True)
        inp = _make_input(ds, use_trajectories=True)

        base_date = _DATES[2]
        dates = [_DATES[0], _DATES[1]]
        current_state = {"date": base_date, "step": np.timedelta64(0, "h")}
        state = inp.load_forcings_state(dates=dates, current_state=current_state)

        for vi, var in enumerate(VARS):
            expected = np.stack([data[0, vi, 0, 0, :], data[1, vi, 0, 0, :]], axis=0)
            np.testing.assert_array_equal(state["fields"][var], expected)
