# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.inference.inputs.utils import convert_dates_to_base_and_step


@pytest.mark.parametrize(
    ("dates", "base", "steps"),
    [
        pytest.param(["2024-06-01T00", "2024-06-01T06"], "2024-06-01T00", [0, 6], id="base date with set"),
        (["2024-06-01T12:00:00", "2024-06-01T18:00:00"], "2024-06-01T06:00:00", [6, 12]),
    ],
)
def test_convert_dates_to_base_and_step(dates, base, steps):
    assert convert_dates_to_base_and_step(dates, base) == steps
