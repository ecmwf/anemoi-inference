# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.metadata import Metadata
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata


@pytest.mark.parametrize(
    "initial, patch, expected",
    [
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"something": {"else": "123"}}}},
            {"dataset": "abc", "something": {"else": "123"}},
        ),
        (
            {"config": {"dataloader": [{"dataset": "abc"}, {"dataset": "xyz"}]}},
            {"config": {"dataloader": {"cutout": [{"dataset": "123"}, {"dataset": "456"}]}}},
            {"cutout": [{"dataset": "123"}, {"dataset": "456"}]},
        ),
        (
            {"config": {"dataloader": "abc"}},
            {"config": {"dataloader": "xyz"}},
            "xyz",
        ),
    ],
)
def test_patch(initial, patch, expected):
    metadata = Metadata(initial)
    assert metadata._metadata == initial

    metadata.patch(patch)
    assert metadata._metadata["config"]["dataloader"] == expected


def test_constant_fields_patch():
    metadata = Metadata(mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False))
    fields = ["z", "sdor", "slor", "lsm"]
    metadata.patch({"dataset": {"constant_fields": fields}})
    assert metadata._metadata["dataset"]["constant_fields"] == fields
