# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Tests for the ``accumulate_from_start_of_forecast`` post-processor."""

from types import SimpleNamespace

import numpy as np

from anemoi.inference.post_processors.accumulate import Accumulate


def _make() -> Accumulate:
    metadata = SimpleNamespace(dataset_name="data", accumulations=["tp"])
    return Accumulate(SimpleNamespace(), metadata, accumulations=["tp"])


def test_accumulates_from_zero():
    acc = _make()
    out = [acc.process({"fields": {"tp": np.full(3, v, dtype=np.float32)}})["fields"]["tp"] for v in (1, 2, 3)]
    assert [o[0] for o in out] == [1, 3, 6]


def test_returned_array_is_not_modified_by_later_steps():
    """The array handed to the output must not be the internal accumulator (outputs may keep a reference)."""
    acc = _make()
    first = acc.process({"fields": {"tp": np.ones(3, dtype=np.float32)}})["fields"]["tp"]
    acc.process({"fields": {"tp": np.ones(3, dtype=np.float32)}})
    acc.process({"fields": {"tp": np.ones(3, dtype=np.float32)}})
    assert list(first) == [1, 1, 1]


def test_negative_values_are_clipped():
    acc = _make()
    out = acc.process({"fields": {"tp": np.array([-1.0, 2.0], dtype=np.float32)}})["fields"]["tp"]
    assert list(out) == [0, 2]
