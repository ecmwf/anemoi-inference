# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``mask`` pre-processor."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from anemoi.metadata import Metadata as CoreMetadata

from anemoi.inference.metadata import InferenceMetadata
from anemoi.inference.pre_processors.mask import MaskValues
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata

N = 6


@pytest.fixture
def metadata() -> InferenceMetadata:
    """The metadata of a dataset with a coordinate array and a mask in its supporting arrays."""
    raw = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    arrays = {
        "latitudes": np.arange(N, dtype=float),
        "longitudes": np.arange(N, dtype=float),
        "my_mask": np.arange(N) % 2,
    }
    return InferenceMetadata(CoreMetadata.from_dict(raw, arrays))


def test_mask_found_in_the_supporting_arrays(metadata: InferenceMetadata):
    processor = MaskValues(MagicMock(), metadata, mask="my_mask", param="2t", mask_value=1)
    assert processor.dataset_name == metadata.dataset_name
    assert processor.filter is not None


def test_mask_from_a_npy_file(metadata: InferenceMetadata, tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.arange(N) % 2)
    processor = MaskValues(MagicMock(), metadata, mask=str(path), param="2t", mask_value=1)
    assert processor.filter is not None


def test_unknown_mask_raises(metadata: InferenceMetadata):
    with pytest.raises(ValueError, match="'nope' not found in supporting arrays"):
        MaskValues(MagicMock(), metadata, mask="nope", param="2t", mask_value=1)
