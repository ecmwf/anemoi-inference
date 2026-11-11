# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import pytest

from anemoi.inference.runners.temporal_downscaler import TemporalDownscalerMultiOutRunner


@pytest.mark.parametrize(
    "target_explicit_times",
    [
        pytest.param([1, 2, 3, 4, 5, 6], id="right-boundary"),
        pytest.param([1, 2, 3, 4, 5], id="no-boundary"),
        pytest.param([0, 1, 2, 3, 4, 5], id="left-boundary"),
    ],
)
def test_temporal_downscaler_stepper_target_steps(target_explicit_times: list[int]) -> None:
    runner = object.__new__(TemporalDownscalerMultiOutRunner)
    runner._checkpoint = SimpleNamespace(input_explicit_times=[0, 6], target_explicit_times=target_explicit_times)
    runner.temporal_downscaling_window = datetime.timedelta(hours=6)
    start = datetime.datetime(2017, 1, 9, 6)

    expected_steps = [datetime.timedelta(hours=t) for t in target_explicit_times]
    assert list(runner.temporal_downscaler_stepper(start)) == [(step, start + step) for step in expected_steps]
