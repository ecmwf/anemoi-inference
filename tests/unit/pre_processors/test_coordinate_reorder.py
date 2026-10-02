# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Tests for the ``reorder`` pre-processor.

Reproduces a real-world grid point-order mismatch: two extracts of the same grid
store the points in a different order --
  * the "target" grid (the model grid) starts each latitude row at longitude 0.0;
  * the "source" grid (the input grid) has each row rolled to start near ~335 deg
    and labels the prime meridian as 360.0 instead of 0.0.
Both describe the *same* physical points, so a per-point field must be reordered
before it is fed to the model.
"""

from typing import cast

import earthkit.data as ekd
import numpy as np
import pytest
from pytest_mock import MockerFixture

from anemoi.inference.metadata import Metadata
from anemoi.inference.pre_processors.coordinate_reorder import CoordinateReorder
from anemoi.inference.pre_processors.coordinate_reorder import build_grid_reordering
from anemoi.inference.pre_processors.coordinate_reorder import invert_reordering


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _source_and_target_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a small (target, source) grid pair plus per-point signals.

    Returns
    -------
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal
        ``target_signal`` is a value per physical point in target (model) order;
        ``source_signal`` is the *same* per-point value in source (input) order.
    """
    lats = np.array([10.0, 0.0, -10.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0])

    target_lat = np.repeat(lats, lons.size)
    target_lon = np.tile(lons, lats.size)
    target_signal = target_lat * 1000.0 + target_lon

    # source: roll each row by 1 (start at 270) and relabel the prime meridian 0 -> 360.
    source_lat = target_lat.copy()
    source_lon = np.tile(np.roll(lons, 1), lats.size)
    source_lon[np.isclose(source_lon, 0.0)] = 360.0

    source_signal = np.empty_like(target_signal)
    for i in range(source_lat.size):
        j = np.where(
            np.isclose(target_lat, source_lat[i]) & np.isclose(np.mod(target_lon, 360.0), np.mod(source_lon[i], 360.0))
        )[0][0]
        source_signal[i] = target_signal[j]

    return target_lat, target_lon, source_lat, source_lon, target_signal, source_signal


def _fieldlist(values, lat, lon, param="2t"):
    """Build an earthkit fieldlist with explicit lat/lon geography."""
    return ekd.from_source(
        "list-of-dicts",
        [
            {
                "param": param,
                "values": np.asarray(values),
                "latitudes": lat,
                "longitudes": lon,
            }
        ],
    )


def _make_reorder(mocker: MockerFixture, model_lat, model_lon, decimals: int = 4) -> CoordinateReorder:
    """Create a CoordinateReorder pre-processor targeting the given model grid."""
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.dataset_name = "data"
    metadata.latitudes = model_lat
    metadata.longitudes = model_lon
    return CoordinateReorder(mocker.MagicMock(), metadata, decimals=decimals)


# --------------------------------------------------------------------------- #
# build_grid_reordering
# --------------------------------------------------------------------------- #
def test_build_grid_reordering_identity_returns_none():
    """Identical grids need no reordering."""
    target_lat, target_lon, *_ = _source_and_target_grids()
    assert build_grid_reordering(target_lat, target_lon, target_lat, target_lon) is None


def test_build_grid_reordering_fixes_rolled_and_360_labelled_grid():
    """Rolled/360-labelled input must be reorderable onto the model grid."""
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal = _source_and_target_grids()

    # The bug: source data used positionally is misaligned with the model grid.
    assert not np.array_equal(source_lon, target_lon)
    assert not np.array_equal(source_signal, target_signal)

    perm = build_grid_reordering(source_lat, source_lon, target_lat, target_lon)
    assert perm is not None

    # The fix: applying the permutation restores the correct per-point values.
    np.testing.assert_array_equal(source_signal[perm], target_signal)
    np.testing.assert_array_equal(source_lat[perm], target_lat)
    circular = np.abs(((source_lon[perm] - target_lon + 180.0) % 360.0) - 180.0)
    assert np.all(circular < 1e-6)


def test_build_grid_reordering_raises_when_not_a_bijection():
    """Grids that do not describe the same point set cannot be reordered."""
    with pytest.raises(ValueError, match="does not match the model grid"):
        build_grid_reordering(
            np.array([0.0, 0.0]),
            np.array([10.0, 20.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 30.0]),
        )


def test_build_grid_reordering_raises_on_size_mismatch():
    """Different numbers of points is an immediate error."""
    with pytest.raises(ValueError, match="source has 1 points, target has 2"):
        build_grid_reordering(
            np.array([0.0]),
            np.array([10.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 20.0]),
        )


# --------------------------------------------------------------------------- #
# invert_reordering
# --------------------------------------------------------------------------- #
def test_invert_reordering_round_trips():
    """The inverse permutation maps target order back to source order."""
    target_lat, target_lon, source_lat, source_lon, *_ = _source_and_target_grids()
    perm = build_grid_reordering(source_lat, source_lon, target_lat, target_lon)
    inverse = invert_reordering(perm)

    assert inverse is not None
    np.testing.assert_array_equal(inverse[perm], np.arange(perm.size))

    source = np.arange(perm.size)  # arbitrary source-ordered data
    target = source[perm]
    np.testing.assert_array_equal(target[inverse], source)


def test_invert_reordering_identity_is_none():
    """The identity mapping (None) is its own inverse."""
    assert invert_reordering(None) is None


def test_invert_reordering_raises_on_non_bijection():
    """A non-permutation index array cannot be inverted."""
    with pytest.raises(ValueError, match="not a bijection|out-of-range"):
        invert_reordering(np.array([0, 0, 1]))


# --------------------------------------------------------------------------- #
# CoordinateReorder.process (end-to-end with earthkit fields)
# --------------------------------------------------------------------------- #
def test_reorder_process_fixes_fields_and_coords(mocker: MockerFixture):
    """An input state on the source grid is reordered onto the model grid."""
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal = _source_and_target_grids()
    processor = _make_reorder(mocker, target_lat, target_lon)

    state = {
        "latitudes": source_lat.copy(),
        "longitudes": source_lon.copy(),
        "fields": _fieldlist(source_signal, source_lat, source_lon),
    }
    # Before: misaligned with the model grid.
    assert not np.array_equal(state["longitudes"], target_lon)

    new_state = processor.process(state)

    # After: the state adopts the *target* (model) coordinates exactly -- not the
    # reordered source values -- so it byte-matches the model grid (this collapses
    # the 360.0 vs 0.0 seam). Field carries correct per-point values.
    np.testing.assert_array_equal(new_state["latitudes"], target_lat)
    np.testing.assert_array_equal(new_state["longitudes"], target_lon)
    field = new_state["fields"][0]
    np.testing.assert_array_equal(field.to_numpy().flatten(), target_signal)
    # The rebuilt earthkit field's geography is consistent with its values.
    grid_lat, grid_lon = field.grid_points()
    np.testing.assert_array_equal(grid_lat, target_lat)
    np.testing.assert_array_equal(grid_lon, target_lon)


def test_reorder_process_collapses_360_vs_0_seam(mocker: MockerFixture):
    """The reordered longitudes use the target's 0.0, never the source's 360.0.

    A strict downstream check (e.g. ``np.allclose(state['longitudes'],
    metadata.longitudes)``) would otherwise fail by ~360 at the seam.
    """
    target_lat, target_lon, source_lat, source_lon, _, source_signal = _source_and_target_grids()
    # The source labels the prime meridian 360.0; the target uses 0.0.
    assert np.any(source_lon == 360.0)
    assert not np.any(target_lon == 360.0)

    processor = _make_reorder(mocker, target_lat, target_lon)
    state = {
        "latitudes": source_lat.copy(),
        "longitudes": source_lon.copy(),
        "fields": _fieldlist(source_signal, source_lat, source_lon),
    }
    new_state = processor.process(state)

    # No spurious 360.0 remains, and a strict positional allclose against the
    # model grid passes (the tensors.py safety-net check).
    assert not np.any(new_state["longitudes"] == 360.0)
    assert np.allclose(new_state["longitudes"], target_lon)
    assert np.allclose(new_state["latitudes"], target_lat)


def test_reorder_process_noop_when_already_aligned(mocker: MockerFixture):
    """A state already on the model grid is returned unchanged (identity perm)."""
    target_lat, target_lon, _, _, target_signal, _ = _source_and_target_grids()
    processor = _make_reorder(mocker, target_lat, target_lon)

    state = {
        "latitudes": target_lat.copy(),
        "longitudes": target_lon.copy(),
        "fields": _fieldlist(target_signal, target_lat, target_lon),
    }
    new_state = processor.process(state)

    assert processor.permutation is None
    np.testing.assert_array_equal(new_state["latitudes"], target_lat)
    np.testing.assert_array_equal(new_state["fields"][0].to_numpy().flatten(), target_signal)


def test_reorder_process_exposes_permutation_and_inverse(mocker: MockerFixture):
    """After processing, the permutation and its inverse are available."""
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal = _source_and_target_grids()
    processor = _make_reorder(mocker, target_lat, target_lon)

    state = {
        "latitudes": source_lat.copy(),
        "longitudes": source_lon.copy(),
        "fields": _fieldlist(source_signal, source_lat, source_lon),
    }
    new_state = processor.process(state)

    perm = processor.permutation
    inverse = processor.inverse_permutation()
    assert perm is not None and inverse is not None

    # The inverse maps the model-ordered field back to the original input order.
    reordered = new_state["fields"][0].to_numpy().flatten()
    np.testing.assert_array_equal(reordered[inverse], source_signal)


def test_reorder_process_raises_on_incompatible_grid(mocker: MockerFixture):
    """A state whose points are not a subset/superset of the model grid errors."""
    target_lat, target_lon, _, _, target_signal, _ = _source_and_target_grids()
    processor = _make_reorder(mocker, target_lat, target_lon)

    # Shift one longitude so the point sets no longer match.
    bad_lon = target_lon.copy()
    bad_lon[0] = 45.0
    state = {
        "latitudes": target_lat.copy(),
        "longitudes": bad_lon,
        "fields": _fieldlist(target_signal, target_lat, bad_lon),
    }
    with pytest.raises(ValueError, match="does not match the model grid"):
        processor.process(state)
