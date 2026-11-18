# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``night_mask`` post-processor, a wrapper of the anemoi-transform ``night_mask`` filter."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from anemoi.transform.filters import filter_registry

from anemoi.inference.post_processors.earthkit_state import wrap_state
from anemoi.inference.post_processors.night_mask import NightMask

pytestmark = pytest.mark.skipif(
    "night_mask" not in {name.replace("-", "_") for name in filter_registry.factories},  # names are normalised
    reason="requires an anemoi-transform that provides the night_mask filter",
)

H = datetime.timedelta(hours=1)
# Equinox: the sun is over the equator, local solar noon is at about 11:53 UTC at 0E.
EQUINOX = datetime.datetime(2016, 9, 22)
LAT = np.zeros(4)
LON = np.array([0.0, 90.0, 180.0, -90.0])
VARIABLES = {name: SimpleNamespace(grib_keys={"param": name}) for name in ("ssr", "ssrd", "2t")}


def _metadata(data_frequency: str = "1h") -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="data",
        data_frequency=data_frequency,
        typed_variables=VARIABLES,
        default_namer=lambda: (lambda field, metadata: metadata["param"]),
    )


def _state(hours: float | None, **extra) -> dict:
    """State with radiation and temperature fields on a 4 point equatorial grid, valid ``hours`` after the equinox."""
    state = {
        "date": EQUINOX + (hours or 0) * H,
        "latitudes": LAT,
        "longitudes": LON,
        "fields": {
            "ssr": np.ones(4, dtype=np.float32),
            "ssrd": np.full(4, 2.0, dtype=np.float32),
            "2t": np.full(4, 280.0, dtype=np.float32),
        },
    }
    if hours is not None:
        state["step"] = hours * H
    state.update(extra)
    return state


def _make(**kwargs) -> NightMask:
    kwargs.setdefault("variables", ["ssr", "ssrd"])
    return NightMask(SimpleNamespace(), _metadata(), **kwargs)


def test_masks_where_dark_at_both_ends() -> None:
    """At 19 UTC the sun is down at 0E for 18:00 and 19:00, and up at 180E (local midday)."""
    out = _make().process(_state(19))["fields"]

    assert out["ssr"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert out["ssrd"].tolist() == [0.0, 0.0, 2.0, 2.0]


def test_hour_after_sunset_is_kept() -> None:
    """At 18 UTC the sun set at 0E within the hour: lit at 17:00, dark at 18:00. The accumulation is kept."""
    out = _make().process(_state(18))["fields"]

    assert out["ssr"][0] == 1.0


def test_other_fields_untouched() -> None:
    state = _state(19)
    out = _make().process(state)

    assert out["fields"]["2t"].tolist() == [280.0] * 4
    assert state["fields"]["ssr"].tolist() == [1.0] * 4  # the input state is not modified
    assert set(out["fields"]) == {"ssr", "ssrd", "2t"}


@pytest.mark.parametrize("hours", [0, None])
def test_initial_state_is_skipped(hours) -> None:
    """Step 0 holds no accumulation: it is left alone, also when the state has no step at all."""
    state = _state(hours)
    out = _make().process(state)["fields"]

    assert out["ssr"].tolist() == [1.0] * 4


def test_initial_state_can_be_masked() -> None:
    out = _make(skip_initial_state=False).process(_state(0))["fields"]

    assert out["ssr"][0] == 0.0  # midnight at 0E
    assert out["ssr"][2] == 1.0  # midday at 180E


def test_does_not_depend_on_the_runner() -> None:
    """No flag of the temporal downscaler runner is needed: a plain state is masked."""
    state = _state(19)
    assert "temporally_downscaled" not in state

    assert _make().process(state)["fields"]["ssr"][0] == 0.0


def test_accumulation_period_defaults_to_the_data_frequency() -> None:
    default = NightMask(SimpleNamespace(), _metadata("6h"), variables=["ssr"])
    explicit = NightMask(SimpleNamespace(), _metadata("6h"), variables=["ssr"], accumulation_period="1h")

    assert default.filter.accumulation_period == "6h"
    assert explicit.filter.accumulation_period == "1h"


def test_wrapped_state_fields_expose_valid_date_and_grid() -> None:
    """What the transform filters need from a state field."""
    field = wrap_state(_state(19), VARIABLES)[0]

    assert field.metadata("valid_datetime") == EQUINOX + 19 * H
    latlon = field.to_latlon(flatten=True)
    assert latlon["lat"].tolist() == LAT.tolist()
    assert latlon["lon"].tolist() == LON.tolist()
