# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
from earthkit.data.readers.grib.codes import GribCodesHandle
from earthkit.data.utils.dates import to_datetime
from earthkit.data.utils.dates import to_timedelta

from anemoi.inference.grib.encoding import encode_message
from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.encoding import render_template
from anemoi.inference.testing.variables import tp
from anemoi.inference.testing.variables import w_100
from anemoi.inference.testing.variables import z


def _make_template(missing_value=9999):
    """Create a minimal GRIB2 template handle from eccodes samples."""
    import eccodes

    raw = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
    eccodes.codes_set(raw, "missingValue", missing_value)
    handle = GribCodesHandle(raw, None, None)

    class _Template:
        def __init__(self, h):
            self.handle = h

    return _Template(handle)


def _read_values(handle):
    """Return the full ndp values array from a CodesHandle (missing→missingValue fill)."""
    import eccodes

    return eccodes.codes_get_double_array(handle._handle, "values")


@pytest.mark.parametrize(
    "missing_value, expect_bitmap",
    [
        (9999, True),
        (-9999, True),
    ],
)
def test_encode_message_nan_becomes_bitmap(missing_value, expect_bitmap):
    """NaN values should be encoded as bitmap-missing, not as real data."""

    template = _make_template(missing_value=missing_value)
    ndp = template.handle.get("numberOfDataPoints")

    values = np.full(ndp, 8500.0)
    values[0] = np.nan  # one NaN that should become bitmap-missing

    handle = encode_message(
        values=values,
        template=template,
        metadata={},
        check_nans=True,
        missing_value=missing_value,
    )

    nv = handle.get("numberOfValues")
    bitmap_missing = ndp - nv
    assert bitmap_missing == 1, f"Expected 1 bitmap-missing point, got {bitmap_missing}"

    out_vals = _read_values(handle)
    mv = handle.get("missingValue")
    present = out_vals[out_vals != mv]
    assert np.all(present == pytest.approx(8500.0)), "Non-NaN values should be unchanged"


def test_encode_message_default_missing_value_is_negative_sentinel():
    """Default missing_value (-9999) avoids collision with physically valid geophysical values."""

    template = _make_template(missing_value=-9999)
    ndp = template.handle.get("numberOfDataPoints")

    values = np.full(ndp, 8500.0)
    values[0] = np.nan

    handle = encode_message(
        values=values,
        template=template,
        metadata={},
        check_nans=True,
        # missing_value not passed — should default to -9999
    )

    nv = handle.get("numberOfValues")
    assert ndp - nv == 1


def test_encode_message_missing_value_collision():
    """When a real value equals the missing_value sentinel, it appears as bitmap-missing.

    This documents the known collision bug: if missing_value=9999 and a grid point
    has a physically valid value that quantizes to exactly 9999, it will be incorrectly
    encoded as bitmap-missing.  Using a large sentinel (e.g. 9.999e20) avoids this.
    """

    template = _make_template(missing_value=9999)
    ndp = template.handle.get("numberOfDataPoints")

    values = np.full(ndp, 8500.0)
    values[0] = np.nan  # intended missing
    values[1] = 9999.0  # real value equal to sentinel — will also become missing

    handle = encode_message(
        values=values,
        template=template,
        metadata={},
        check_nans=True,
        missing_value=9999,
    )

    nv = handle.get("numberOfValues")
    bitmap_missing = ndp - nv
    # Both the NaN and the 9999 real value are treated as missing
    assert bitmap_missing == 2, f"Expected 2 bitmap-missing (NaN + real 9999 collision), got {bitmap_missing}"


def test_encode_message_negative_sentinel_no_collision():
    """Using a negative sentinel (-9999) avoids collision with real geophysical values."""

    NEGATIVE_SENTINEL = -9999
    template = _make_template(missing_value=NEGATIVE_SENTINEL)
    ndp = template.handle.get("numberOfDataPoints")

    values = np.full(ndp, 8500.0)
    values[0] = np.nan  # intended missing
    values[1] = 9999.0  # real value equal to old positive sentinel — should NOT become missing

    handle = encode_message(
        values=values,
        template=template,
        metadata={},
        check_nans=True,
        missing_value=NEGATIVE_SENTINEL,
    )

    nv = handle.get("numberOfValues")
    bitmap_missing = ndp - nv
    assert bitmap_missing == 1, f"Expected exactly 1 bitmap-missing (only the NaN), got {bitmap_missing}"


@pytest.mark.parametrize(
    "template, handle, expected",
    [
        (
            "{dateTime}_{step:03}.grib",
            {"dateTime": "202001011200", "step": 6},
            "202001011200_006.grib",
        ),
        (
            "{validityDate}{validityTime:04}.grib",
            {"validityDate": "20200101", "validityTime": 900},
            "202001010900.grib",
        ),
    ],
)
def test_render_template(template, handle, expected):
    """Test the render_template function.

    NOTE: we mock the handle with a dictionary.
    """
    result = render_template(template, handle)
    assert result == expected


@pytest.mark.parametrize(
    "variable, date, step, start_steps, expected_keys",
    [
        (
            tp,
            to_datetime("20250101T0000"),
            to_timedelta(0),
            {},
            {
                "date": 20241231,
                "time": 1800,
                "startStep": 0,
                "endStep": 6,
                "stepType": "accum",
                "shortName": "tp",
                "dataType": "fc",
            },
        ),
        (
            tp,
            to_datetime("20250101T0000"),
            to_timedelta(6),
            {},
            {
                "date": 20250101,
                "time": 0,
                "startStep": 0,
                "endStep": 6,
                "stepType": "accum",
                "shortName": "tp",
                "dataType": "fc",
            },
        ),
        (
            tp,
            to_datetime("20250101T0000"),
            to_timedelta(12),
            {},
            {
                "date": 20250101,
                "time": 0,
                "startStep": 6,
                "endStep": 12,
                "stepType": "accum",
                "shortName": "tp",
                "dataType": "fc",
            },
        ),
        (
            tp,
            to_datetime("20250101T0000"),
            to_timedelta(12),
            {"tp": to_timedelta(0)},
            {
                "date": 20250101,
                "time": 0,
                "startStep": 0,
                "endStep": 12,
                "stepType": "accum",
                "shortName": "tp",
                "dataType": "fc",
            },
        ),
        (
            z,
            to_datetime("20250101T0000"),
            to_timedelta(0),
            {},
            {
                "date": 20250101,
                "time": 0,
                "step": 0,
                "stepType": "instant",
                "shortName": "z",
                "dataType": "fc",
            },
        ),
        (
            z,
            to_datetime("20250101T0000"),
            to_timedelta(6),
            {},
            {
                "date": 20250101,
                "time": 0,
                "step": 6,
                "stepType": "instant",
                "shortName": "z",
                "dataType": "fc",
            },
        ),
        (
            w_100,
            to_datetime("20250101T0000"),
            to_timedelta(6),
            {},
            {
                "date": 20250101,
                "time": 0,
                "step": 6,
                "stepType": "instant",
                "shortName": "w",
                "dataType": "fc",
                "levelist": 100,
                "typeOfLevel": "isobaricInhPa",
            },
        ),
    ],
)
def test_grib_keys(variable, date, step, start_steps, expected_keys):
    encoding = grib_keys(
        values=None,
        template=None,
        variable=variable,
        ensemble=False,
        param=variable.param,
        date=date,
        step=step,
        previous_step=None,
        start_steps=start_steps,
        keys={},
    )

    for key, expected_value in expected_keys.items():
        assert encoding[key] == expected_value
