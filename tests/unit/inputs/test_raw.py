# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the raw input.

These tests write raw ``.npz`` files with :class:`RawOutput` and read them
back with :class:`RawInput`, verifying that a model can consume the raw output
of another model as its initial conditions (stacking).
"""

from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from anemoi.inference.inputs.raw import RawInput
from anemoi.inference.outputs.raw import RawOutput

ALL_VARIABLES = ("z_500", "cp", "2t")

LAGGED_DATES = [datetime(2019, 12, 31, 18), datetime(2020, 1, 1, 0)]

GRID = np.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch):
    """Stub the environment scan, which otherwise dominates the runtime of these tests."""
    monkeypatch.setattr("anemoi.inference.outputs.raw.gather_provenance_info", lambda: {})


@pytest.fixture
def context():
    """A mock context for testing the raw input/output."""
    ctx = MagicMock()
    ctx.reference_date = datetime(2020, 1, 1, 0)
    ctx.typed_variables = {}
    ctx.output_frequency = None
    ctx.write_initial_state = False
    return ctx


@pytest.fixture
def metadata():
    """A mock metadata for testing the raw input/output."""
    meta = MagicMock()
    meta.dataset_name = "test"
    meta.typed_variables = {name: MagicMock() for name in ALL_VARIABLES}
    meta.multi_dataset = False
    meta.grid = "o48"
    return meta


def _write_raw_files(context, metadata, directory, *, with_grid=True, **kwargs):
    """Write raw files (one per lagged date) mimicking a first model's output."""
    output = RawOutput(context, metadata, dir=str(directory), output_manifest=True, **kwargs)

    grid = {"latitudes": GRID, "longitudes": GRID}
    output.open({"date": LAGGED_DATES[0], **(grid if with_grid else {})})

    expected = {}
    for date in LAGGED_DATES:
        fields = {name: np.arange(4, dtype=np.float32) + i for i, name in enumerate(ALL_VARIABLES)}
        # make each date's values distinct so stacking order can be checked
        fields = {name: values + date.hour for name, values in fields.items()}
        expected[date] = fields
        output.write_step({"date": date, "step": date - LAGGED_DATES[0], "fields": fields, **grid})
    return expected


def test_raw_roundtrip_stacking(tmp_path, context, metadata):
    """RawInput stacks the raw output of RawOutput along the date dimension."""
    expected = _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    state = input_.create_input_state(dates=LAGGED_DATES)

    # reference date is the last (most recent) date
    assert state["date"] == LAGGED_DATES[-1]
    assert sorted(state["fields"]) == sorted(ALL_VARIABLES)

    for name in ALL_VARIABLES:
        values = state["fields"][name]
        assert values.shape == (len(LAGGED_DATES), 4)
        for i, date in enumerate(sorted(LAGGED_DATES)):
            np.testing.assert_array_equal(values[i], expected[date][name])

    np.testing.assert_array_equal(state["latitudes"], GRID)
    np.testing.assert_array_equal(state["longitudes"], GRID)
    assert sorted(state["_variables"]) == sorted(ALL_VARIABLES)
    assert state["_input"] is input_


def test_raw_variable_selection(tmp_path, context, metadata):
    """RawInput only exposes the requested variables."""
    _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=["cp"])
    state = input_.create_input_state(dates=LAGGED_DATES)

    assert sorted(state["fields"]) == ["cp"]


def test_raw_conflicting_variable_raises(tmp_path, context, metadata):
    """RawInput raises if the configured variables are not in the manifest."""
    _write_raw_files(context, metadata, tmp_path)

    with pytest.raises(ValueError, match="not in the manifest"):
        RawInput(context, metadata, dir=str(tmp_path), variables=["does_not_exist"])


def test_raw_missing_manifest_raises(tmp_path, context, metadata):
    """RawInput raises when the directory holds no manifest."""
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))


def test_raw_missing_file_raises(tmp_path, context, metadata):
    """RawInput raises a clear error when a file for a date is missing."""
    _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    with pytest.raises(FileNotFoundError, match="no raw file found"):
        input_.create_input_state(dates=[datetime(2021, 1, 1, 0)])


def test_raw_load_forcings_state(tmp_path, context, metadata):
    """RawInput can load a forcings state for a given set of dates."""
    _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    current_state = {"date": LAGGED_DATES[-1], "step": timedelta(0)}
    state = input_.load_forcings_state(dates=LAGGED_DATES, current_state=current_state)

    assert state["dates"] == sorted(LAGGED_DATES)
    for name in ALL_VARIABLES:
        assert state["fields"][name].shape == (len(LAGGED_DATES), 4)


def test_raw_stacking_order_independent_of_input_order(tmp_path, context, metadata):
    """Fields are always stacked in ascending date order regardless of input order."""
    expected = _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    # Pass the dates in reverse order.
    state = input_.create_input_state(dates=list(reversed(LAGGED_DATES)))

    # The most recent date is still the reference date.
    assert state["date"] == max(LAGGED_DATES)
    for name in ALL_VARIABLES:
        values = state["fields"][name]
        for i, date in enumerate(sorted(LAGGED_DATES)):
            np.testing.assert_array_equal(values[i], expected[date][name])


def test_raw_single_date(tmp_path, context, metadata):
    """RawInput works with a single requested date."""
    expected = _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    date = LAGGED_DATES[-1]
    state = input_.create_input_state(dates=[date])

    for name in ALL_VARIABLES:
        values = state["fields"][name]
        assert values.shape == (1, 4)
        np.testing.assert_array_equal(values[0], expected[date][name])


def test_raw_custom_template_from_manifest(tmp_path, context, metadata):
    """A custom filename convention is recovered from the manifest, not configured."""
    _write_raw_files(context, metadata, tmp_path, template="state_{date}.npz", strftime="%Y-%m-%dT%H")

    assert (tmp_path / "state_2020-01-01T00.npz").exists()

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    state = input_.create_input_state(dates=LAGGED_DATES)

    for name in ALL_VARIABLES:
        assert state["fields"][name].shape == (len(LAGGED_DATES), 4)


@pytest.mark.parametrize("with_grid", [True, False], ids=["from_grid_npz", "from_step_file"])
def test_raw_reference_coordinates(with_grid, tmp_path, context, metadata):
    """Coordinates come from grid.npz when written, and from a step file otherwise."""
    _write_raw_files(context, metadata, tmp_path, with_grid=with_grid)

    assert (tmp_path / "grid.npz").exists() is with_grid

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    np.testing.assert_array_equal(input_.latitudes, GRID)
    np.testing.assert_array_equal(input_.longitudes, GRID)
