# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers to build synthetic earthkit-data fields from MARS-style metadata.

Several places in anemoi-inference need to turn plain arrays plus a variable's
GRIB keys into earthkit-data fields (the dummy input, the state wrapper used by
the transform filters, ...). earthkit-data 1.0 builds fields from *components*
rather than from a flat metadata dictionary, so the MARS-style keys have to be
dispatched to the right component. This module is the single place where that
translation happens.
"""

import datetime
import logging
from collections.abc import Iterator
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import earthkit.data as ekd

from anemoi.inference.types import FloatArray

LOG = logging.getLogger(__name__)

# Types that earthkit-data can store verbatim as labels.
LABEL_TYPES = (str, int, float, bool, datetime.datetime, datetime.timedelta)


@lru_cache(maxsize=1)
def _levtype_to_level_type() -> dict[str, str]:
    """Map MARS level type abbreviations to earthkit-data level type names.

    earthkit-data identifies level types by *name* ("surface", "pressure", ...)
    and silently registers any unknown string as a brand new level type. MARS
    abbreviations ("sfc", "pl", ...) must therefore be translated before being
    handed to ``Field.from_components``.

    Returns
    -------
    dict[str, str]
        Mapping of MARS abbreviation to earthkit-data level type name.
    """
    # earthkit-data does not re-export its level type registry anywhere public, so
    # this reads it from its defining module. Deriving the mapping rather than
    # hardcoding it means it cannot drift out of step with earthkit-data, and if the
    # module is ever moved the ImportError is immediate and test-covered rather than
    # silently producing bogus level types.
    from earthkit.data.field.component.level_type import LevelTypes

    return {t.value.abbreviation: t.value.name for t in LevelTypes}


def field_from_grib_keys(
    values: FloatArray,
    grib_keys: dict[str, Any],
    *,
    name: str,
    valid_datetime: Any | None = None,
    geography: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
) -> ekd.Field:
    """Build an earthkit-data field from values and a variable's GRIB keys.

    The GRIB keys are exposed twice: verbatim as labels, so that a MARS-style
    lookup returns exactly the configured value, and — for the keys earthkit-data
    models explicitly — as proper components, so that consumers using the
    component API (such as ``anemoi.transform.variables.Variable.from_earthkit``)
    can find them.

    Parameters
    ----------
    values : FloatArray
        The field values.
    grib_keys : dict[str, Any]
        MARS-style GRIB keys, as returned by ``Variable.grib_keys``.
    name : str
        The anemoi variable name, stored as the ``name`` label.
    valid_datetime : Any, optional
        The valid datetime of the field.
    geography : dict[str, Any], optional
        The geography component, e.g. ``{"latitudes": ..., "longitudes": ...}``.
    labels : dict[str, Any], optional
        Extra labels to attach to the field.

    Returns
    -------
    ekd.Field
        The created field.
    """
    all_labels: dict[str, Any] = {k: v for k, v in grib_keys.items() if isinstance(v, LABEL_TYPES)}
    all_labels.update(labels or {})
    all_labels["name"] = name

    vertical: dict[str, Any] = {}
    if (levelist := grib_keys.get("levelist")) is not None:
        vertical["level"] = levelist
    if (levtype := grib_keys.get("levtype")) is not None:
        # Unknown abbreviations are left out rather than registered as new level types.
        if (level_type := _levtype_to_level_type().get(levtype)) is not None:
            vertical["level_type"] = level_type
        else:
            LOG.debug("No earthkit-data level type for levtype %r, leaving it as a label only.", levtype)

    return ekd.Field.from_components(
        values=values,
        parameter={"variable": grib_keys.get("param", name)},
        time={"valid_datetime": valid_datetime} if valid_datetime is not None else None,
        geography=geography,
        vertical=vertical or None,
        labels=all_labels,
    )


# Keys exposed when a `FieldMetadata` is iterated over. Any other key can still be
# looked up explicitly; this list only controls iteration/`dict()`/`repr()`.
_COMMON_METADATA_KEYS = (
    "param",
    "levelist",
    "levtype",
    "number",
    "step",
    "valid_datetime",
    "base_datetime",
)


class FieldMetadata(Mapping):
    """A MARS-style, read-only view of an earthkit-data field's metadata.

    earthkit-data 1.0 replaced the flat metadata dictionary that used to be handed
    to namer functions with namespaced components. This mapping restores the old
    flat lookup: keys are resolved on demand as raw (GRIB) metadata keys, then as
    earthkit-data component keys, then as user labels.

    Lookups are lazy so that namers configured with arbitrary keys (see
    `RulesNamer`) keep working, as they did when they received the full GRIB
    metadata dictionary.

    Parameters
    ----------
    field : Any
        The field to read the metadata from.
    """

    _MISSING = object()

    def __init__(self, field: Any) -> None:
        self._field = field
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        value = self._cache.get(key, self._MISSING)
        if value is self._MISSING:
            value = self._cache[key] = self._resolve(key)
        if value is self._MISSING:
            raise KeyError(key)
        return value

    def _resolve(self, key: str) -> Any:
        from anemoi.transform.metadata import get_metadata

        if key == "levtype":
            return self._resolve_levtype()

        value = get_metadata(self._field, key, default=None)
        return self._MISSING if value is None else value

    def _resolve_levtype(self) -> Any:
        """Resolve `levtype` as a MARS-style abbreviation.

        The component API exposes earthkit-data's level type *name* ("surface",
        "pressure", ...), whereas namers expect the MARS abbreviation ("sfc",
        "pl", ...). `vertical.abbreviation` provides the latter. The raw metadata
        key is still preferred, as it carries values (such as "o2d") that
        earthkit-data has no level type for.
        """
        candidates = (
            lambda: self._field.metadata("levtype"),
            lambda: self._field.get("vertical.abbreviation"),
            lambda: self._field.get("labels.levtype"),
        )
        for candidate in candidates:
            try:
                levtype = candidate()
            except (KeyError, TypeError, AttributeError):
                continue
            if levtype is not None and levtype != "unknown":
                return levtype
        return self._MISSING

    def __iter__(self) -> Iterator[str]:
        return (key for key in _COMMON_METADATA_KEYS if key in self)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"


def get_metadata_dict(field: Any) -> Mapping:
    """Build a MARS-style metadata mapping from a field, for use with namer functions.

    Parameters
    ----------
    field : Any
        The field to extract metadata from.

    Returns
    -------
    Mapping
        A lazily-resolved mapping of MARS-style metadata key to value.
    """
    return FieldMetadata(field)


def name_fields(data: Any, namer: callable) -> Any:
    """Apply a namer function to all fields and set labels.name.

    Parameters
    ----------
    data : Any
        The fieldlist to name.
    namer : callable
        The namer function: (field, metadata_dict) -> str.

    Returns
    -------
    Any
        A new fieldlist with labels.name set on each field.
    """
    named = []
    for f in data:
        md = get_metadata_dict(f)
        name = namer(f, md)
        named.append(f.set(**{"labels.name": name}))
    return ekd.create_fieldlist(named)
