# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest

pytest_plugins = "anemoi.utils.testing"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--longtests",
        action="store_true",
        dest="longtests",
        default=False,
        help="enable tests marked as longtests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'longtests' marker to avoid warnings."""
    config.addinivalue_line("markers", "longtests: mark tests as long-running")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip @pytest.mark.longtests tests unless --longtests is used."""
    if not config.getoption("--longtests"):
        skip_marker = pytest.mark.skip(reason="Skipping long test, use --longtests to enable")
        for item in items:
            if item.get_closest_marker("longtests"):
                item.add_marker(skip_marker)


STATE_NPOINTS = 50


@pytest.fixture
def state():
    """Fixture to create a mock state for testing."""

    return {
        "latitudes": np.random.uniform(-90, 90, size=STATE_NPOINTS),
        "longitudes": np.random.uniform(-180, 180, size=STATE_NPOINTS),
        "fields": {
            "2t": np.random.uniform(250, 310, size=STATE_NPOINTS),
            "z_850": np.random.uniform(500, 1500, size=STATE_NPOINTS),
        },
        "date": datetime(2020, 1, 1, 0, 0),
        "step": timedelta(hours=6),
    }


@pytest.fixture
def extract_mask_npy(tmp_path):
    """Fixture to create a mock mask in .npy format."""
    mask = np.random.choice([True, False], size=STATE_NPOINTS)
    mask_path = tmp_path / "extract_mask.npy"
    np.save(mask_path, mask)
    return str(mask_path)


@pytest.fixture
def assign_mask_npy(tmp_path):
    """Fixture to create a mock mask in .npy format."""
    mask = np.zeros(STATE_NPOINTS + 10, dtype=bool)
    mask[:STATE_NPOINTS] = True
    mask_path = tmp_path / "assign_mask.npy"
    np.save(mask_path, mask)
    return str(mask_path)
