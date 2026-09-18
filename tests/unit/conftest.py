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

ALL_VARIABLES = ("z_500", "cp", "2t")


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
    metadata.typed_variables = {name: MagicMock() for name in ALL_VARIABLES}
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
