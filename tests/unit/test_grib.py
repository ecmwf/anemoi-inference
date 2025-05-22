import sys

import pytest

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
