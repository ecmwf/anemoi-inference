# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys

import pytest
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import as_timedelta

from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.encoding import render_template
from anemoi.inference.testing.variables import tp
from anemoi.inference.testing.variables import w_100
from anemoi.inference.testing.variables import z


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Format specifier requires python 3.10 or higher.",
)
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
            as_datetime("20250101T0000"),
            as_timedelta(0),
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
            as_datetime("20250101T0000"),
            as_timedelta(6),
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
            as_datetime("20250101T0000"),
            as_timedelta(12),
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
            as_datetime("20250101T0000"),
            as_timedelta(12),
            {"tp": as_timedelta(0)},
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
            as_datetime("20250101T0000"),
            as_timedelta(0),
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
            as_datetime("20250101T0000"),
            as_timedelta(6),
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
            as_datetime("20250101T0000"),
            as_timedelta(6),
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
