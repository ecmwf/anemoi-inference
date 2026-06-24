# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from earthkit.data.utils.dates import to_datetime
from earthkit.data.utils.dates import to_timedelta

from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.encoding import render_template
from anemoi.inference.testing.variables import lsm_with_paramid
from anemoi.inference.testing.variables import lsm_without_paramid
from anemoi.inference.testing.variables import tp
from anemoi.inference.testing.variables import w_100
from anemoi.inference.testing.variables import z


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
                "shortName": "tp",  # this variable doesn't have paramId in the metadata, so we expect shortName out
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
        (
            lsm_with_paramid,
            to_datetime("20250101T0000"),
            to_timedelta(0),
            {},
            {
                "date": 20250101,
                "time": 0,
                "step": 0,
                "stepType": "instant",
                "paramId": 172,  # this variable has paramId in the metadata, so we expect paramId out
                "dataType": "fc",
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
        date=date,
        step=step,
        previous_step=None,
        start_steps=start_steps,
        keys={},
    )

    for key, expected_value in expected_keys.items():
        assert encoding[key] == expected_value


@pytest.mark.parametrize(
    "convert_grib_paramid, expected_keys",
    [
        (
            False,
            {"shortName": "lsm"},
        ),
        (
            True,
            {"paramId": 172},
        ),
    ],
)
def test_grib_keys_convert_paramid(convert_grib_paramid, expected_keys):
    encoding = grib_keys(
        values=None,
        template=None,
        variable=lsm_without_paramid,
        ensemble=False,
        date=to_datetime("20250101T0000"),
        step=to_timedelta(0),
        previous_step=None,
        start_steps={},
        keys={},
        convert_grib_paramid=convert_grib_paramid,
    )

    for key, expected_value in expected_keys.items():
        assert encoding[key] == expected_value

    assert "shortName" not in encoding if "paramId" in expected_keys else "paramId" not in encoding
