# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests that state fields carry what date- and grid-dependent anemoi-transform filters need."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from anemoi.transform.filters import filter_registry

from anemoi.inference.post_processors import create_post_processor
from anemoi.inference.post_processors.earthkit_state import wrap_state

H = datetime.timedelta(hours=1)
# Equinox: the sun is over the equator, local solar noon is at about 11:53 UTC at 0E.
EQUINOX = datetime.datetime(2016, 9, 22)
LAT = np.zeros(4)
LON = np.array([0.0, 90.0, 180.0, -90.0])
VARIABLES = {name: SimpleNamespace(grib_keys={"param": name}) for name in ("ssr", "ssrd", "2t")}

requires_night_mask = pytest.mark.skipif(
    "night_mask" not in {name.replace("-", "_") for name in filter_registry.factories},  # names are normalised
    reason="requires an anemoi-transform that provides the night_mask filter",
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


def _post_processor(**filter_options):
    metadata = SimpleNamespace(
        dataset_name="data",
        typed_variables=VARIABLES,
        default_namer=lambda: (lambda field, metadata: metadata["param"]),
    )
    config = {"forward_transform_filter": dict(**filter_options)}
    return create_post_processor(SimpleNamespace(), config, metadata)


def test_wrapped_state_fields_expose_valid_date_and_grid() -> None:
    """What date- and grid-dependent transform filters read from a field."""
    field = wrap_state(_state(19), VARIABLES)[0]

    assert field.metadata("valid_datetime") == EQUINOX + 19 * H
    latlon = field.to_latlon(flatten=True)
    assert latlon["lat"].tolist() == LAT.tolist()
    assert latlon["lon"].tolist() == LON.tolist()


def test_valid_datetime_follows_the_state_date() -> None:
    first = wrap_state(_state(1), VARIABLES)[0].metadata("valid_datetime")
    later = wrap_state(_state(7), VARIABLES)[0].metadata("valid_datetime")
    assert later - first == 6 * H


@requires_night_mask
def test_night_mask_filter_through_forward_transform_filter() -> None:
    processor = _post_processor(
        filter={"night_mask": {"variables": ["ssr", "ssrd"], "accumulation_period": "1h"}},
        skip_initial_state=True,
    )

    # 19:00 UTC: dark at 18:00 and 19:00 over 0E and 90E, lit over 180E and -90E.
    out = processor.process(_state(19))["fields"]

    assert out["ssr"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert out["ssrd"].tolist() == [0.0, 0.0, 2.0, 2.0]
    assert out["2t"].tolist() == [280.0] * 4  # not listed in `variables`


@requires_night_mask
def test_night_mask_keeps_the_hour_after_sunset() -> None:
    processor = _post_processor(
        filter={"night_mask": {"variables": ["ssr"], "accumulation_period": "1h"}},
        skip_initial_state=True,
    )

    # 18:00 UTC: the sun set over 0E at about 17:53, so the window (17:00, 18:00] had sunlight.
    assert processor.process(_state(18))["fields"]["ssr"][0] == 1.0


@requires_night_mask
@pytest.mark.parametrize("hours", [None, 0])
def test_initial_state_is_skipped_with_skip_initial_state(hours) -> None:
    """An initial state has no accumulation window and holds real data: it must stay unchanged."""
    processor = _post_processor(
        filter={"night_mask": {"variables": ["ssr"], "accumulation_period": "1h"}},
        skip_initial_state=True,
    )
    state = _state(hours)
    assert processor.process(state) is state
