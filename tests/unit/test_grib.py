import sys

import pytest
from anemoi.transform.variables import Variable
from anemoi.utils.dates import as_timedelta

from anemoi.inference.grib.encoding import encode_time_processing
from anemoi.inference.grib.encoding import render_template


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


tp = Variable.from_dict(
    "tp",
    {
        "mars": {
            "param": "tp",
            "levtype": "sfc",
        },
        "process": "accumulation",
        "period": [0, 6],
    },
)
z = Variable.from_dict(
    "z",
    {
        "mars": {
            "param": "z",
            "levtype": "sfc",
        }
    },
)


@pytest.mark.parametrize(
    "variable, date, time, step, expected_keys",
    [
        (
            tp,
            20250101,
            0,
            as_timedelta(0),
            {"date": 20241231, "time": 1800, "startStep": 0, "endStep": 6, "stepType": "accum"},
        ),
        (
            tp,
            20250101,
            0,
            as_timedelta(6),
            {"date": 20250101, "time": 0, "startStep": 0, "endStep": 6, "stepType": "accum"},
        ),
        (z, 20250101, 0, as_timedelta(0), {"date": 20250101, "time": 0, "step": 0, "stepType": "instant"}),
    ],
)
def test_encode_time_processing(variable, date, time, step, expected_keys):
    encoding = {"date": date, "time": time}

    encode_time_processing(
        result=encoding,
        template=None,
        variable=variable,
        step=step,
        previous_step=None,
        start_steps={},
        edition=1,
        ensemble=False,
    )

    for key, expected_value in expected_keys.items():
        assert encoding[key] == expected_value
