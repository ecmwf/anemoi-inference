# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..processor import Processor
from . import pre_processor_registry

LOG = logging.getLogger(__name__)


def build_grid_reordering(
    src_lat: FloatArray,
    src_lon: FloatArray,
    tgt_lat: FloatArray,
    tgt_lon: FloatArray,
    decimals: int = 4,
) -> IntArray | None:
    """Build a permutation that maps a source grid onto a target grid ordering.

    The source and target grids may store the same physical points in a
    different order (for example, longitude rows rolled to start at a different
    meridian, or the prime meridian labelled ``360.0`` instead of ``0.0``). The
    mapping is resolved by matching on rounded ``(lat, lon)`` coordinates, so it
    works for any permutation rather than assuming a simple roll. Longitudes are
    compared modulo 360 so that ``360.0`` and ``0.0`` are treated as the same
    meridian.

    Parameters
    ----------
    src_lat, src_lon : FloatArray
        Coordinates of the source points (the input state grid), 1-D.
    tgt_lat, tgt_lon : FloatArray
        Coordinates of the target points (the model/metadata grid), 1-D.
    decimals : int
        Rounding used when matching coordinates.

    Returns
    -------
    IntArray | None
        ``None`` if the grids are already in the same order (identity mapping).
        Otherwise an integer index array ``perm`` of length ``len(tgt_lat)`` such
        that ``src_field[perm]`` is aligned to the target order.

    Raises
    ------
    ValueError
        If the two grids do not describe the same set of points (i.e. the
        mapping is not a bijection): points are missing from one side.
    """
    src_lat = np.asarray(src_lat).ravel()
    src_lon = np.asarray(src_lon).ravel()
    tgt_lat = np.asarray(tgt_lat).ravel()
    tgt_lon = np.asarray(tgt_lon).ravel()

    if src_lat.size != tgt_lat.size:
        raise ValueError(
            f"Cannot reorder input grid onto model grid: source has {src_lat.size} "
            f"points, target has {tgt_lat.size} points"
        )

    def _canonical_lon(lon: FloatArray) -> FloatArray:
        lon_r = np.round(np.mod(lon, 360.0), decimals)
        # Collapse the 360.0 == 0.0 edge that can survive rounding.
        return np.where(np.isclose(lon_r, 360.0), 0.0, lon_r)

    s_lat = np.round(src_lat, decimals)
    s_lon = _canonical_lon(src_lon)
    t_lat = np.round(tgt_lat, decimals)
    t_lon = _canonical_lon(tgt_lon)

    # Fast path: coordinates already identical element-wise -> no reordering.
    if np.array_equal(s_lat, t_lat) and np.array_equal(s_lon, t_lon):
        return None

    from collections import defaultdict
    from collections import deque

    src_index: dict[tuple[float, float], "deque[int]"] = defaultdict(deque)
    for i in range(src_lat.size):
        src_index[(s_lat[i], s_lon[i])].append(i)

    perm = np.full(tgt_lat.size, -1, dtype=np.int64)
    used = np.zeros(src_lat.size, dtype=bool)
    matched = 0
    for t in range(tgt_lat.size):
        q = src_index.get((t_lat[t], t_lon[t]))
        if q:
            i = q.popleft()
            perm[t] = i
            used[i] = True
            matched += 1

    if matched != tgt_lat.size or matched != src_lat.size:
        n_target_only = int(np.count_nonzero(perm < 0))
        n_source_only = int(np.count_nonzero(~used))
        raise ValueError(
            "Input state grid does not match the model grid: matched "
            f"{matched} of target={tgt_lat.size}, source={src_lat.size} points "
            f"({n_target_only} target-only, {n_source_only} source-only). "
            "The grids must describe the same set of points to be reordered."
        )

    # If the permutation is the identity, signal 'no reorder needed'.
    if np.array_equal(perm, np.arange(tgt_lat.size)):
        return None

    return perm


def invert_reordering(perm: IntArray | None) -> IntArray | None:
    """Return the inverse of a reordering permutation.

    If ``perm`` maps a source array onto a target ordering via
    ``target = source[perm]``, then the returned ``inverse`` maps back via
    ``source = target[inverse]`` (equivalently ``inverse[perm] == arange``). This
    is useful to map a model-ordered result back onto the original input ordering.

    Parameters
    ----------
    perm : IntArray | None
        A permutation index array (as returned by :func:`build_grid_reordering`),
        or ``None`` for the identity mapping.

    Returns
    -------
    IntArray | None
        ``None`` if ``perm`` is ``None`` (identity is its own inverse). Otherwise
        the inverse permutation, with the same length and dtype as ``perm``.

    Raises
    ------
    ValueError
        If ``perm`` is not a valid permutation of ``range(len(perm))`` (contains
        negative/out-of-range indices or duplicates), which would make it
        non-invertible.
    """
    if perm is None:
        return None

    perm = np.asarray(perm)
    n = perm.size

    if perm.ndim != 1:
        raise ValueError(f"Permutation must be 1-D, got shape {perm.shape}")
    if n and (perm.min() < 0 or perm.max() >= n):
        raise ValueError("Permutation contains out-of-range indices; it is not invertible")

    inverse = np.empty(n, dtype=perm.dtype)
    inverse[perm] = np.arange(n, dtype=perm.dtype)

    # Validate it really was a bijection (no duplicate targets left a gap).
    if not np.array_equal(inverse[perm], np.arange(n, dtype=perm.dtype)):
        raise ValueError("Permutation is not a bijection; it is not invertible")

    return inverse


@pre_processor_registry.register("coordinate_reorder")
class CoordinateReorder(Processor):
    """Reorder the input state onto the model coordinate ordering.

    Some input sources store the same physical grid in a different point order than the
    model expects -- for example longitude rows rolled to start at a different
    meridian, or the prime meridian labelled ``360.0`` instead of ``0.0``. Such a
    state is misaligned when fed to the model.

    This pre-processor matches the state's ``(latitudes, longitudes)`` against the model grid
    (``metadata.latitudes``/``longitudes``, treating ``360.0`` as ``0.0``) and
    permutes the coordinates and every field so they are consistent with the
    model ordering.

    If the two grids are already in the same order the state is returned
    unchanged. If they do not describe the same set of points a ``ValueError`` is
    raised.
    """

    # The permutation applied by the last ``process`` call (``None`` if the
    # state was already aligned and no reordering was needed).
    _perm: IntArray | None = None

    def __init__(self, context: Context, metadata: Metadata, *, decimals: int = 4) -> None:
        """Initialize the CoordinateReorder processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        metadata : Metadata
            Metadata corresponding to the dataset this processor is handling.
        decimals : int
            Rounding used when matching source and target coordinates.
        """
        super().__init__(context, metadata)
        self.decimals = decimals

    def __repr__(self) -> str:
        return f"CoordinateReorder(decimals={self.decimals})"

    @property
    def permutation(self) -> IntArray | None:
        """The permutation applied by the last :meth:`process` call.

        ``target = source[permutation]``. ``None`` means no reordering was needed
        (identity), or :meth:`process` has not been called yet.
        """
        return self._perm

    def inverse_permutation(self) -> IntArray | None:
        """Return the inverse of the permutation applied by :meth:`process`.

        Maps a model-ordered array back onto the original input ordering via
        ``source = target[inverse]``. Returns ``None`` if no reordering was
        applied (identity is its own inverse).
        """
        return invert_reordering(self.permutation)

    def process(self, state: State) -> State:
        """Reorder the state onto the model grid ordering.

        Parameters
        ----------
        state : State
            The state containing ``latitudes``, ``longitudes`` and ``fields``.

        Returns
        -------
        State
            The reordered state (or the original state if already aligned).
        """
        src_lat = state.get("latitudes")
        src_lon = state.get("longitudes")
        if src_lat is None or src_lon is None:
            raise ValueError("Input state must contain 'latitudes' and 'longitudes'")

        tgt_lat = self.metadata.latitudes
        tgt_lon = self.metadata.longitudes

        if tgt_lat is None or tgt_lon is None:
            LOG.warning(
                "[%s] Reorder pre-processor: model grid has no latitudes/longitudes; " "skipping.",
                self.dataset_name,
            )
            return state

        perm = build_grid_reordering(src_lat, src_lon, tgt_lat, tgt_lon, decimals=self.decimals)
        self._perm = perm
        if perm is None:
            # Already in the correct order.
            return state

        LOG.info(
            "[%s] Reordering input state grid to match the model grid ordering (%d points)",
            self.dataset_name,
            perm.size,
        )

        # The permutation aligns the source points to exactly the target grid
        # points (``build_grid_reordering`` guarantees a bijection, matching mod
        # 360). Adopt the *target* (model) coordinates so the result is canonical
        # and byte-matches ``metadata`` -- in particular this collapses the
        # ``360.0`` vs ``0.0`` seam that reordering the source values would leave
        # behind (a 360 deg discrepancy no tolerance could reconcile downstream).
        new_lat = np.asarray(tgt_lat).ravel()
        new_lon = np.asarray(tgt_lon).ravel()

        state = state.copy()
        result = []
        for field in state["fields"]:
            data = field.to_numpy()[..., perm]
            # Rebuild as an earthkit array field: set the reordered data, then the
            # reordered geography so the field's grid points stay consistent with
            # its values (e.g. for downstream ``grid_points()``/``to_latlon()``).
            new_field = new_field_from_numpy(data, template=field)
            new_field = new_field_from_latitudes_longitudes(new_field, new_lat, new_lon)
            result.append(new_field)

        state["fields"] = new_fieldlist_from_list(result)
        state["latitudes"] = new_lat
        state["longitudes"] = new_lon

        state["_coordinate_reorder"] = {
            "permutation": self.inverse_permutation(),
            "latitudes": src_lat,
            "longitudes": src_lon,
        }
        return state
