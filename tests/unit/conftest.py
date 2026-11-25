# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

# Load torch before healpy (pulled in by anemoi-transform filters): on macOS both ship an
# OpenMP runtime, and torch crashes if healpy's is loaded first.
import torch  # noqa: F401
from anemoi.metadata import Metadata as CoreMetadata

from anemoi.inference.testing.mock_checkpoint import mock_load_metadata
from anemoi.inference.testing.variables import cp
from anemoi.inference.testing.variables import two_t
from anemoi.inference.testing.variables import z_500

TYPED_VARIABLES = {v.name: v for v in (z_500, cp, two_t)}
ALL_VARIABLES = tuple(TYPED_VARIABLES)


@pytest.fixture
def atmos_core() -> CoreMetadata:
    """The `anemoi.metadata.Metadata` of the legacy single-dataset test checkpoint (no arrays)."""
    return CoreMetadata.from_dict(mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False))


@pytest.fixture
def multi_core() -> CoreMetadata:
    """The `anemoi.metadata.Metadata` of the multi-dataset test checkpoint (one dataset, `data`; no arrays)."""
    return CoreMetadata.from_dict(mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False))


@pytest.fixture
def basic_context():
    """Fixture to create a mock context for testing outputs."""
    context = MagicMock()
    context.reference_date = "2020-01-01"
    context.typed_variables = {}
    context.output_frequency = None
    context.write_initial_state = False
    return context


@pytest.fixture
def basic_metadata():
    """Fixture to create mock metadata for testing outputs."""
    metadata = MagicMock()
    metadata.dataset_name = "test"
    metadata.dataset_view.data.typed_variables.return_value = dict(TYPED_VARIABLES)
    metadata.multi_dataset = False
    return metadata


@pytest.fixture
def basic_state():
    """Fixture to create a mock state for testing outputs."""
    return {
        "date": datetime(2020, 1, 1),
        "step": timedelta(hours=6),
        "fields": {name: np.array([1.0]) for name in ALL_VARIABLES},
        "latitudes": np.array([0.0]),
        "longitudes": np.array([0.0]),
    }
