# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Tests for the ``undo_coordinate_reorder`` post-processor.

The ``coordinate_reorder`` pre-processor permutes an input state onto the model
grid ordering and records what it did under ``state["_coordinate_reorder"]``.
The ``undo_coordinate_reorder`` post-processor reverses that operation so the
output is written back in the *original* input ordering.

These tests exercise the two guarantees that matter:
  * a full round-trip (pre-processor reorder -> post-processor undo) restores the
    original per-point values and coordinates;
  * when the pre-processor needed no reordering (grids already aligned), the
    post-processor is a no-op.
"""

from typing import cast

import earthkit.data as ekd
import numpy as np
from pytest_mock import MockerFixture

from anemoi.inference.metadata import Metadata
from anemoi.inference.post_processors.coordinate_reorder import UndoCoordinateReorder
from anemoi.inference.pre_processors.coordinate_reorder import CoordinateReorder


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


def _make_undo(mocker: MockerFixture) -> UndoCoordinateReorder:
    """Create an UndoCoordinateReorder post-processor."""
    metadata = cast(Metadata, mocker.MagicMock())
    metadata.dataset_name = "data"
    return UndoCoordinateReorder(mocker.MagicMock(), metadata)


# --------------------------------------------------------------------------- #
# Round-trip: reorder (pre) then undo (post) restores the input ordering
# --------------------------------------------------------------------------- #
def test_undo_reorder_round_trips(mocker: MockerFixture):
    """Pre-processor reorder followed by post-processor undo is the identity.

    The pre-processor maps the source-ordered state onto the model grid (and
    records the inverse permutation in ``_coordinate_reorder``). The
    post-processor must map it back so the final per-point values and
    coordinates match the original source state exactly.
    """
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal = _source_and_target_grids()

    pre = _make_reorder(mocker, target_lat, target_lon)
    reordered = pre.process(
        {
            "latitudes": source_lat.copy(),
            "longitudes": source_lon.copy(),
            "fields": _fieldlist(source_signal, source_lat, source_lon),
        }
    )

    # Sanity: the pre-processor actually reordered onto the model grid.
    assert pre.forward_mapping is not None
    np.testing.assert_array_equal(reordered["latitudes"], target_lat)
    np.testing.assert_array_equal(reordered["longitudes"], target_lon)

    # The post-processor consumes a state whose fields are a name -> ndarray map
    # (as produced downstream of the runner), carrying the reorder bookkeeping.
    post_state = {
        "latitudes": np.asarray(reordered["latitudes"]).copy(),
        "longitudes": np.asarray(reordered["longitudes"]).copy(),
        "fields": {"2t": reordered["fields"][0].to_numpy().flatten()},
        "_coordinate_reorder": reordered["_coordinate_reorder"],
    }

    undo = _make_undo(mocker)
    final = undo.process(post_state)

    # After undo: back in the original source ordering, bit-for-bit.
    np.testing.assert_array_equal(final["latitudes"], source_lat)
    np.testing.assert_array_equal(final["longitudes"], source_lon)
    np.testing.assert_array_equal(final["fields"]["2t"], source_signal)


def test_undo_reorder_round_trips_multidimensional_fields(mocker: MockerFixture):
    """The undo reorders along the last (grid) axis for multi-dim fields."""
    target_lat, target_lon, source_lat, source_lon, target_signal, source_signal = _source_and_target_grids()

    pre = _make_reorder(mocker, target_lat, target_lon)
    reordered = pre.process(
        {
            "latitudes": source_lat.copy(),
            "longitudes": source_lon.copy(),
            "fields": _fieldlist(source_signal, source_lat, source_lon),
        }
    )
    perm = pre.forward_mapping
    assert perm is not None

    # Build a (time, points) field in model order and stash it as an ndarray.
    model_ordered = source_signal[perm]
    field_2d = np.stack([model_ordered, model_ordered + 1.0])
    post_state = {
        "latitudes": np.asarray(reordered["latitudes"]).copy(),
        "longitudes": np.asarray(reordered["longitudes"]).copy(),
        "fields": {"2t": field_2d.copy()},
        "_coordinate_reorder": reordered["_coordinate_reorder"],
    }

    undo = _make_undo(mocker)
    final = undo.process(post_state)

    expected = np.stack([source_signal, source_signal + 1.0])
    np.testing.assert_array_equal(final["fields"]["2t"], expected)
    np.testing.assert_array_equal(final["latitudes"], source_lat)
    np.testing.assert_array_equal(final["longitudes"], source_lon)


# --------------------------------------------------------------------------- #
# No-op: when the pre-processor did not reorder, the undo changes nothing
# --------------------------------------------------------------------------- #
def test_undo_reorder_is_noop_when_no_reorder_was_applied(mocker: MockerFixture):
    """If the input grid already matched the model grid, no reorder bookkeeping is
    recorded, so the post-processor leaves the state untouched.
    """
    target_lat, target_lon, _, _, target_signal, _ = _source_and_target_grids()

    pre = _make_reorder(mocker, target_lat, target_lon)
    reordered = pre.process(
        {
            "latitudes": target_lat.copy(),
            "longitudes": target_lon.copy(),
            "fields": _fieldlist(target_signal, target_lat, target_lon),
        }
    )

    # Pre-processor detected the grids were already aligned: no reorder, and no
    # ``_coordinate_reorder`` bookkeeping added to the state.
    assert pre.forward_mapping is None
    assert "_coordinate_reorder" not in reordered

    post_state = {
        "latitudes": target_lat.copy(),
        "longitudes": target_lon.copy(),
        "fields": {"2t": target_signal.copy()},
    }

    undo = _make_undo(mocker)
    final = undo.process(post_state)

    # Nothing moved.
    np.testing.assert_array_equal(final["latitudes"], target_lat)
    np.testing.assert_array_equal(final["longitudes"], target_lon)
    np.testing.assert_array_equal(final["fields"]["2t"], target_signal)


def test_undo_reorder_is_noop_without_bookkeeping(mocker: MockerFixture):
    """Directly: a state lacking ``_coordinate_reorder`` is returned unchanged."""
    target_lat, target_lon, _, _, target_signal, _ = _source_and_target_grids()

    post_state = {
        "latitudes": target_lat.copy(),
        "longitudes": target_lon.copy(),
        "fields": {"2t": target_signal.copy()},
    }

    undo = _make_undo(mocker)
    final = undo.process(post_state)

    np.testing.assert_array_equal(final["latitudes"], target_lat)
    np.testing.assert_array_equal(final["longitudes"], target_lon)
    np.testing.assert_array_equal(final["fields"]["2t"], target_signal)


def test_undo_reorder_is_noop_when_permutation_is_none(mocker: MockerFixture):
    """A ``_coordinate_reorder`` entry with a ``None`` permutation is a no-op."""
    target_lat, target_lon, _, _, target_signal, _ = _source_and_target_grids()

    post_state = {
        "latitudes": target_lat.copy(),
        "longitudes": target_lon.copy(),
        "fields": {"2t": target_signal.copy()},
        "_coordinate_reorder": {"permutation": None},
    }

    undo = _make_undo(mocker)
    final = undo.process(post_state)

    np.testing.assert_array_equal(final["latitudes"], target_lat)
    np.testing.assert_array_equal(final["longitudes"], target_lon)
    np.testing.assert_array_equal(final["fields"]["2t"], target_signal)


# --------------------------------------------------------------------------- #
# repr
# --------------------------------------------------------------------------- #
def test_undo_reorder_repr(mocker: MockerFixture):
    assert repr(_make_undo(mocker)) == "UndoCoordinateReorder()"
