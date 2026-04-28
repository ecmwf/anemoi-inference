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


STATE_NPOINTS = 50


def pytest_addoption(parser):
    parser.addoption("--cosmo", action="store_true", default=False, help="only run cosmo tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "cosmo: mark test as requiring cosmo eccodes definitions and isolation")


def pytest_collection_modifyitems(config, items):
    skip_cosmo = pytest.mark.skip(reason="skipping cosmo tests, use --cosmo to run")
    skip_non_cosmo = pytest.mark.skip(reason="skipping non-cosmo tests")

    for item in items:
        if config.getoption("--cosmo"):
            if "cosmo" not in item.keywords:
                item.add_marker(skip_non_cosmo)
        else:
            if "cosmo" in item.keywords:
                item.add_marker(skip_cosmo)


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
