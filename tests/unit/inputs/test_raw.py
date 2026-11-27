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

import threading
import time
from collections import defaultdict
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


def _write_raw_files(context, metadata, directory, *, with_grid=True, delay=0, **kwargs):
    """Write raw files (one per lagged date) mimicking a first model's output."""
    output = RawOutput(context, metadata, dir=str(directory), **kwargs)

    grid = {"latitudes": GRID, "longitudes": GRID}
    output.open({"date": LAGGED_DATES[0], **(grid if with_grid else {})})

    expected = {}
    for date in LAGGED_DATES:
        time.sleep(delay)
        fields = {name: np.arange(4, dtype=np.float32) + i for i, name in enumerate(ALL_VARIABLES)}
        # make each date's values distinct so stacking order can be checked
        fields = {name: values + date.hour for name, values in fields.items()}
        expected[date] = fields
        output.write_step({"date": date, "step": date - LAGGED_DATES[0], "fields": fields, **grid})
    output.close()
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

    # The reader looks for the rendered name as is, so the output refuses a template without `.npz`.
    with pytest.raises(ValueError, match="must end with '.npz'"):
        RawOutput(context, metadata, dir=str(tmp_path / "other"), template="state_{date}")


def test_raw_ref_date_index(tmp_path, context, metadata):
    """The date of the state is taken at `ref_date_index`, as the temporal downscaler asks for the first."""
    expected = _write_raw_files(context, metadata, tmp_path)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    state = input_.create_input_state(dates=LAGGED_DATES, ref_date_index=0)

    assert state["date"] == LAGGED_DATES[0]
    np.testing.assert_array_equal(state["fields"]["cp"][0], expected[LAGGED_DATES[0]]["cp"])

    # The index applies to the sorted dates, so the date always matches its row.
    state = input_.create_input_state(dates=list(reversed(LAGGED_DATES)), ref_date_index=0)
    assert state["date"] == LAGGED_DATES[0]
    np.testing.assert_array_equal(state["fields"]["cp"][0], expected[LAGGED_DATES[0]]["cp"])


def test_raw_wait_for_writer(tmp_path, context, metadata):
    """With `wait`, the directory, the manifest and the files may appear after the input is created.

    The coordinates come from the files that are read, not from any other ``.npz`` in the directory.
    """
    directory = tmp_path / "later"
    input_ = RawInput(
        context, metadata, dir=str(directory), variables=list(ALL_VARIABLES), wait=True, timeout=10, poll_interval=0.01
    )
    assert not directory.exists()

    expected = {}

    def writer():
        time.sleep(0.1)
        directory.mkdir()
        np.savez(directory / "0.npz", latitudes=-GRID, longitudes=-GRID)  # sorts before the step files
        expected.update(_write_raw_files(context, metadata, directory, delay=0.1))

    thread = threading.Thread(target=writer)
    thread.start()
    state = input_.create_input_state(dates=LAGGED_DATES)
    thread.join()

    for name in ALL_VARIABLES:
        for i, date in enumerate(LAGGED_DATES):
            np.testing.assert_array_equal(state["fields"][name][i], expected[date][name])
    np.testing.assert_array_equal(state["latitudes"], GRID)
    np.testing.assert_array_equal(state["longitudes"], GRID)


def test_raw_wait_timeout_and_complete(tmp_path, context, metadata):
    """Waiting for a file raises on timeout, and at once when the writer has finished without it."""
    _write_raw_files(context, metadata, tmp_path)
    later = [datetime(2020, 1, 1, 6)]

    # The writer is still running: only the partial manifest is there.
    (tmp_path / "manifest.json").rename(tmp_path / "manifest.json.part")
    input_ = RawInput(
        context, metadata, dir=str(tmp_path), variables=["cp"], wait=True, timeout=0.05, poll_interval=0.01
    )
    with pytest.raises(TimeoutError):
        input_.create_input_state(dates=later)

    # The writer has finished.
    (tmp_path / "manifest.json.part").rename(tmp_path / "manifest.json")
    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=["cp"], wait=True, timeout=60, poll_interval=0.01)
    start = time.monotonic()
    with pytest.raises(FileNotFoundError, match="no raw file found"):
        input_.create_input_state(dates=later)
    assert time.monotonic() - start < 1

    # Nothing written yet.
    empty = tmp_path / "empty"
    empty.mkdir()
    input_ = RawInput(context, metadata, dir=str(empty), variables=["cp"], wait=True, timeout=0.05, poll_interval=0.01)
    with pytest.raises(TimeoutError, match="manifest.json.part"):
        input_.create_input_state(dates=later)


def test_raw_unfinished_run_needs_wait(tmp_path, context, metadata):
    """Without `wait`, files whose writer has not finished are refused."""
    _write_raw_files(context, metadata, tmp_path)
    (tmp_path / "manifest.json").rename(tmp_path / "manifest.json.part")

    with pytest.raises(FileNotFoundError, match="has not finished: set `wait`"):
        RawInput(context, metadata, dir=str(tmp_path), variables=["cp"])

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=["cp"], wait=True, timeout=1, poll_interval=0.01)
    assert input_.create_input_state(dates=LAGGED_DATES)["fields"]["cp"].shape == (2, 4)


def test_raw_reference_date_check(tmp_path, context, metadata):
    """With `select_reference_date`, the files must have been written for the run's reference date."""
    _write_raw_files(context, metadata, tmp_path)

    other = MagicMock(reference_date=datetime(2020, 1, 2, 0))
    input_ = RawInput(other, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))

    input_.create_input_state(dates=LAGGED_DATES)
    with pytest.raises(ValueError, match="written for reference date"):
        input_.create_input_state(dates=LAGGED_DATES, select_reference_date=True)

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    input_.create_input_state(dates=LAGGED_DATES, select_reference_date=True)


def test_raw_missing_variable_raises(tmp_path, context, metadata):
    """A variable missing from a file, such as a forcing after the initial state, is named in the error."""
    output = RawOutput(context, metadata, dir=str(tmp_path))
    grid = {"latitudes": GRID, "longitudes": GRID}
    output.open({"date": LAGGED_DATES[0], **grid})
    for i, date in enumerate(LAGGED_DATES):
        names = ALL_VARIABLES if i == 0 else ALL_VARIABLES[:-1]
        fields = {name: np.zeros(4, dtype=np.float32) for name in names}
        output.write_step({"date": date, "step": date - LAGGED_DATES[0], "fields": fields, **grid})
    output.close()

    input_ = RawInput(context, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    with pytest.raises(ValueError, match=r"\['2t'\] are not in .*20200101000000.npz.*constant_forcings"):
        input_.create_input_state(dates=LAGGED_DATES)


def test_raw_reused_directory(tmp_path, context, metadata, monkeypatch):
    """A later run into the same directory rewrites the manifest, so its files are read for its own date."""
    _write_raw_files(context, metadata, tmp_path)

    # The next cycle: same directory, reference date one day later, opened several times like the downscaler does.
    monkeypatch.setattr("anemoi.inference.decorators.UNIQUE_PATHS", defaultdict(set))
    later = MagicMock(reference_date=context.reference_date + timedelta(days=1), typed_variables={})
    dates = [date + timedelta(days=1) for date in LAGGED_DATES]
    output = RawOutput(later, metadata, dir=str(tmp_path))
    grid = {"latitudes": GRID, "longitudes": GRID}
    for date in dates:
        output.open({"date": dates[0], **grid})
        fields = {name: np.full(4, 7, dtype=np.float32) for name in ALL_VARIABLES}
        output.write_step({"date": date, "step": date - dates[0], "fields": fields, **grid})
    output.close()

    input_ = RawInput(later, metadata, dir=str(tmp_path), variables=list(ALL_VARIABLES))
    state = input_.create_input_state(dates=dates, select_reference_date=True, ref_date_index=0)

    assert state["date"] == dates[0]
    np.testing.assert_array_equal(state["fields"]["cp"], np.full((2, 4), 7, dtype=np.float32))
