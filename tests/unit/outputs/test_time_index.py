# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests that NetCDF and Zarr outputs compute the time index from the state
date correctly.

This guarantees that when a single forecast step is delivered through several
``write_step`` calls (as happens when a parallel output chunk strategy produces
more chunks than there are writers, so one writer receives several chunks for
the same step) all the fields for that step land at the same time index, and
the time coordinate is written correctly.

``open()`` receives the step-zero state (date == forecast start); with
``write_initial_state`` disabled (the default here) step zero is not written and
the first forecast step lands at time index 0.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

TIMESTEP = datetime.timedelta(hours=6)
START = datetime.datetime(2020, 1, 1, 0, 0)
N_VALUES = 4
ALL_FIELDS = ["a", "b", "c", "d", "e"]


# ── fixtures / helpers ────────────────────────────────────────────────────────


def _make_context_and_metadata(*, lead_time=None, multi_step_output=1, write_initial_state=False):
    context = MagicMock()
    context.reference_date = START
    context.typed_variables = {}
    context.output_frequency = None
    context.write_initial_state = write_initial_state
    context.allow_nans = True
    context.lead_time = lead_time

    output_offsets = [(s + 1) * TIMESTEP for s in range(multi_step_output)]

    metadata = MagicMock()
    metadata.dataset_name = "test"
    metadata.typed_variables = {}
    metadata.output_offsets = output_offsets
    metadata.rollout_shift = output_offsets[-1]
    metadata.multi_step_output = multi_step_output
    return context, metadata


def _make_state(step_index, field_names=ALL_FIELDS):
    """State at forecast ``step_index``. ``step_index == 0`` is step zero
    (date == forecast start), the state that ``open()`` receives.
    """
    date = START + step_index * TIMESTEP
    return {
        "date": date,
        "step": step_index * TIMESTEP,
        "latitudes": np.arange(N_VALUES, dtype="f4"),
        "longitudes": np.arange(N_VALUES, dtype="f4"),
        "fields": {name: np.full(N_VALUES, float(v), dtype="f4") for v, name in enumerate(field_names)},
    }


def _split_fields(state, n_chunks):
    """Split a state's fields into ``n_chunks`` states (like a chunk strategy)."""
    names = list(state["fields"])
    chunks = []
    for i in range(n_chunks):
        chunk = dict(state)
        chunk["fields"] = {name: state["fields"][name] for name in names[i::n_chunks]}
        chunks.append(chunk)
    return chunks


# ── NetCDF ────────────────────────────────────────────────────────────────────


class TestNetCDFTimeIndex:
    def _make_output(self, path, *, multi_step_output=1, write_initial_state=False):
        pytest.importorskip("netCDF4")
        from anemoi.inference.outputs.netcdf import NetCDFOutput

        context, metadata = _make_context_and_metadata(
            multi_step_output=multi_step_output,
            write_initial_state=write_initial_state,
        )
        return NetCDFOutput(context, metadata, path=str(path))

    def _read(self, path):
        from netCDF4 import Dataset

        return Dataset(path, "r")

    def test_times_written_at_correct_indices(self, tmp_path):
        out = self._make_output(tmp_path / "out.nc")
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        for s in forecast:
            out.write_step(s)
        out.close()

        ds = self._read(tmp_path / "out.nc")
        try:
            periods = ds.variables["forecast_period"][:]
            assert list(periods) == [int(s["step"].total_seconds()) for s in forecast]
            for i, s in enumerate(forecast):
                for name in ALL_FIELDS:
                    np.testing.assert_array_equal(ds.variables[name][i], s["fields"][name])
        finally:
            ds.close()

    def test_split_step_lands_on_single_index(self, tmp_path):
        """A step delivered over several write_step calls must not be scattered."""
        out = self._make_output(tmp_path / "out.nc")
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        for s in forecast:
            # simulate 3 chunks per step (more chunks than writers -> round robin)
            for chunk in _split_fields(s, 3):
                out.write_step(chunk)
        out.close()

        ds = self._read(tmp_path / "out.nc")
        try:
            # exactly one time entry per forecast step (no scattering)
            assert ds.dimensions["time"].size == len(forecast)
            periods = ds.variables["forecast_period"][:]
            assert list(periods) == [int(s["step"].total_seconds()) for s in forecast]
            for i, s in enumerate(forecast):
                for name in ALL_FIELDS:
                    np.testing.assert_array_equal(ds.variables[name][i], s["fields"][name])
        finally:
            ds.close()

    def test_split_matches_whole(self, tmp_path):
        """Writing whole states and split states must produce identical files."""
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]

        whole = self._make_output(tmp_path / "whole.nc")
        whole.open(step0)
        for s in forecast:
            whole.write_step(s)
        whole.close()

        split = self._make_output(tmp_path / "split.nc")
        split.open(step0)
        for s in forecast:
            for chunk in _split_fields(s, 3):
                split.write_step(chunk)
        split.close()

        a = self._read(tmp_path / "whole.nc")
        b = self._read(tmp_path / "split.nc")
        try:
            np.testing.assert_array_equal(a.variables["forecast_period"][:], b.variables["forecast_period"][:])
            np.testing.assert_array_equal(a.variables["time"][:], b.variables["time"][:])
            for name in ALL_FIELDS:
                np.testing.assert_array_equal(a.variables[name][:], b.variables[name][:])
        finally:
            a.close()
            b.close()

    def test_out_of_order_delivery(self, tmp_path):
        """Steps delivered out of order still land at the correct index."""
        out = self._make_output(tmp_path / "out.nc")
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        # deliver forecast steps out of order
        out.write_step(forecast[2])
        out.write_step(forecast[0])
        out.write_step(forecast[1])
        out.close()

        ds = self._read(tmp_path / "out.nc")
        try:
            for i, s in enumerate(forecast):
                for name in ALL_FIELDS:
                    np.testing.assert_array_equal(ds.variables[name][i], s["fields"][name])
        finally:
            ds.close()

    def test_write_step_zero_puts_step0_at_index0(self, tmp_path):
        """With write_initial_state, step zero is written at index 0."""
        out = self._make_output(tmp_path / "out.nc", write_initial_state=True)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 3)]
        out.open(step0)
        out.write_step(step0)  # step zero
        for s in forecast:
            out.write_step(s)
        out.close()

        ds = self._read(tmp_path / "out.nc")
        try:
            periods = ds.variables["forecast_period"][:]
            assert list(periods) == [0] + [int(s["step"].total_seconds()) for s in forecast]
            all_states = [step0] + forecast
            for i, s in enumerate(all_states):
                for name in ALL_FIELDS:
                    np.testing.assert_array_equal(ds.variables[name][i], s["fields"][name])
        finally:
            ds.close()

    def test_multi_step_output_spacing(self, tmp_path):
        """With multi_step_output > 1 the output offsets are [T, 2T]; the step
        spacing used for the index must be the smallest offset (T).
        """
        out = self._make_output(tmp_path / "out.nc", multi_step_output=2)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 5)]
        out.open(step0)
        for s in forecast:
            for chunk in _split_fields(s, 3):
                out.write_step(chunk)
        out.close()

        ds = self._read(tmp_path / "out.nc")
        try:
            assert ds.dimensions["time"].size == len(forecast)
            periods = ds.variables["forecast_period"][:]
            assert list(periods) == [int(s["step"].total_seconds()) for s in forecast]
            for i, s in enumerate(forecast):
                for name in ALL_FIELDS:
                    np.testing.assert_array_equal(ds.variables[name][i], s["fields"][name])
        finally:
            ds.close()


# ── Zarr ──────────────────────────────────────────────────────────────────────


class TestZarrTimeIndex:
    def _make_output(self, store, *, multi_step_output=1, write_initial_state=False, lead_time=None):
        pytest.importorskip("zarr")
        from anemoi.inference.outputs.zarr import ZarrOutput

        context, metadata = _make_context_and_metadata(
            lead_time=lead_time or 4 * TIMESTEP,
            multi_step_output=multi_step_output,
            write_initial_state=write_initial_state,
        )
        metadata.typed_variables = {name: SimpleNamespace(grib_keys={"param": name}) for name in ALL_FIELDS}
        return ZarrOutput(context, metadata, store=str(store))

    def _read(self, store):
        import zarr

        return zarr.open_group(str(store), mode="r")

    def test_times_written_at_correct_indices(self, tmp_path):
        store = tmp_path / "out.zarr"
        out = self._make_output(store)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        for s in forecast:
            out.write_step(s)
        out.close()

        g = self._read(store)
        times = g["time"][:]
        # time value is (date - reference_date) in seconds; reference_date is
        # resolved by ZarrOutput.open() from the context / offsets.
        expected = [int((s["date"] - out.reference_date).total_seconds()) for s in forecast]
        assert list(times[: len(forecast)]) == expected
        for i, s in enumerate(forecast):
            for name in ALL_FIELDS:
                np.testing.assert_array_equal(g[name][i], s["fields"][name])

    def test_split_step_lands_on_single_index(self, tmp_path):
        store = tmp_path / "out.zarr"
        out = self._make_output(store)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        for s in forecast:
            for chunk in _split_fields(s, 3):
                out.write_step(chunk)
        out.close()

        g = self._read(store)
        times = g["time"][:]
        expected = [int((s["date"] - out.reference_date).total_seconds()) for s in forecast]
        # only the first len(forecast) slots are written; the rest stay at fill
        assert list(times[: len(forecast)]) == expected
        for i, s in enumerate(forecast):
            for name in ALL_FIELDS:
                np.testing.assert_array_equal(g[name][i], s["fields"][name])

    def test_split_matches_whole(self, tmp_path):
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]

        whole = self._make_output(tmp_path / "whole.zarr")
        whole.open(step0)
        for s in forecast:
            whole.write_step(s)
        whole.close()

        split = self._make_output(tmp_path / "split.zarr")
        split.open(step0)
        for s in forecast:
            for chunk in _split_fields(s, 3):
                split.write_step(chunk)
        split.close()

        a = self._read(tmp_path / "whole.zarr")
        b = self._read(tmp_path / "split.zarr")
        np.testing.assert_array_equal(a["time"][:], b["time"][:])
        for name in ALL_FIELDS:
            np.testing.assert_array_equal(a[name][:], b[name][:])

    def test_out_of_order_delivery(self, tmp_path):
        store = tmp_path / "out.zarr"
        out = self._make_output(store)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 4)]
        out.open(step0)
        out.write_step(forecast[2])
        out.write_step(forecast[0])
        out.write_step(forecast[1])
        out.close()

        g = self._read(store)
        for i, s in enumerate(forecast):
            for name in ALL_FIELDS:
                np.testing.assert_array_equal(g[name][i], s["fields"][name])

    def test_write_step_zero_puts_step0_at_index0(self, tmp_path):
        store = tmp_path / "out.zarr"
        out = self._make_output(store, write_initial_state=True)
        step0 = _make_state(0)
        forecast = [_make_state(i) for i in range(1, 3)]
        out.open(step0)
        out.write_step(step0)
        for s in forecast:
            out.write_step(s)
        out.close()

        g = self._read(store)
        all_states = [step0] + forecast
        times = g["time"][:]
        expected = [int((s["date"] - out.reference_date).total_seconds()) for s in all_states]
        assert list(times[: len(all_states)]) == expected
        for i, s in enumerate(all_states):
            for name in ALL_FIELDS:
                np.testing.assert_array_equal(g[name][i], s["fields"][name])
