# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``night_mask`` post-processor."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.inference.post_processors.night_mask import NightMask

T0 = datetime.datetime(2016, 9, 7, 6)
H = datetime.timedelta(hours=1)

# insolation per valid hour on 4 grid points:
#   0 always dark, 1 sunset (lit at 6h only), 2 sunrise (lit from 8h), 3 always lit
INSOLATION = {
    T0: np.array([0, 0.3, 0, 0.5], dtype=np.float32),
    T0 + 1 * H: np.array([0, 0, 0, 0.5], dtype=np.float32),
    T0 + 2 * H: np.array([0, 0, 0.1, 0.5], dtype=np.float32),
}


def _make(monkeypatch, **kwargs) -> NightMask:
    metadata = SimpleNamespace(dataset_name="data", data_frequency="1h")
    kwargs.setdefault("variables", ["ssr", "ssrd"])
    processor = NightMask(SimpleNamespace(), metadata, **kwargs)
    calls = []

    def fake_compute(lat, lon, date):
        calls.append(date)
        return INSOLATION[date]

    monkeypatch.setattr(processor, "_compute", fake_compute)
    processor.calls = calls
    return processor


def _state(date, downscaled=True, dtype=np.float32) -> dict:
    return {
        "date": date,
        "temporally_downscaled": downscaled,
        "latitudes": np.array([0.0, 10.0, 20.0, 30.0]),
        "longitudes": np.array([0.0, 10.0, 20.0, 30.0]),
        "fields": {
            "ssr": np.full(4, 5.0, dtype=dtype),
            "ssrd": np.full(4, 7.0, dtype=dtype),
            "2t": np.full(4, 280.0, dtype=dtype),
        },
    }


def test_masks_only_where_dark_at_both_ends(monkeypatch):
    processor = _make(monkeypatch)

    out = processor.process(_state(T0 + 1 * H))["fields"]  # window (6h, 7h]
    # point 0 dark both ends -> 0; point 1 lit at 6h (sunset hour) -> kept; 2 dark both; 3 lit
    assert out["ssr"].tolist() == [0.0, 5.0, 0.0, 5.0]
    assert out["ssrd"].tolist() == [0.0, 7.0, 0.0, 7.0]

    out = processor.process(_state(T0 + 2 * H))["fields"]  # window (7h, 8h]
    # point 1 now dark at both ends -> 0; point 2 lit at 8h (sunrise hour) -> kept
    assert out["ssr"].tolist() == [0.0, 0.0, 5.0, 5.0]


def test_other_fields_untouched_and_input_state_not_modified(monkeypatch):
    processor = _make(monkeypatch)
    state = _state(T0 + 1 * H)

    out = processor.process(state)

    assert out["fields"]["2t"].tolist() == [280.0] * 4
    assert state["fields"]["ssr"].tolist() == [5.0] * 4  # input untouched
    assert out is not state and out["fields"] is not state["fields"]


def test_dtype_preserved(monkeypatch):
    processor = _make(monkeypatch)
    out = processor.process(_state(T0 + 1 * H, dtype=np.float64))["fields"]["ssr"]
    assert out.dtype == np.float64


def test_non_downscaled_states_are_skipped(monkeypatch):
    """Boundary / input states are data, not model output: no masking, no insolation computed."""
    processor = _make(monkeypatch)
    state = _state(T0 + 1 * H, downscaled=False)
    assert processor.process(state) is state
    state.pop("temporally_downscaled")
    assert processor.process(state) is state
    assert processor.calls == []


def test_missing_field_raises(monkeypatch):
    processor = _make(monkeypatch, variables=["ssr", "fdir"])
    with pytest.raises(ValueError, match=r"\['fdir'\]"):
        processor.process(_state(T0 + 1 * H))


def test_mask_field_cache_reuses_previous_date(monkeypatch):
    """The t-1h of one output is the t of the previous one: each date is computed once."""
    processor = _make(monkeypatch)
    processor.process(_state(T0 + 1 * H))
    processor.process(_state(T0 + 2 * H))
    assert processor.calls == [T0 + 1 * H, T0, T0 + 2 * H]


def test_accumulation_period_default_and_override(monkeypatch):
    assert _make(monkeypatch).accumulation_period == H
    assert _make(monkeypatch, accumulation_period="30m").accumulation_period == datetime.timedelta(minutes=30)
    with pytest.raises(ValueError, match="at least one"):
        _make(monkeypatch, variables=[])


def test_insolation_is_computed_with_the_forcings_source():
    """Real computation on a tiny grid: sun over (0N, 0E) at noon UTC on the equinox, dark at midnight."""
    pytest.importorskip("earthkit.data")
    pytest.importorskip("anemoi.transform")
    metadata = SimpleNamespace(dataset_name="data", data_frequency="1h")
    processor = NightMask(SimpleNamespace(), metadata, variables=["ssr"])
    lat, lon = np.array([0.0, 0.0]), np.array([0.0, 180.0])

    noon = processor._compute(lat, lon, datetime.datetime(2016, 9, 22, 12))
    midnight = processor._compute(lat, lon, datetime.datetime(2016, 9, 22, 0))

    assert noon[0] > 0.9 and noon[1] == 0.0
    assert midnight[0] == 0.0 and midnight[1] > 0.9
