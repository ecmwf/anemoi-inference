# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Tests for the earthkit-data field helpers in ``anemoi.inference.fields``."""

import datetime

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi.inference.fields import FieldMetadata
from anemoi.inference.fields import _levtype_to_level_type
from anemoi.inference.fields import field_from_grib_keys
from anemoi.inference.fields import get_metadata_dict
from anemoi.inference.fields import name_fields

DATE = datetime.datetime(2020, 1, 1, 6)
LATS = np.array([10.0, 10.0, 0.0, 0.0])
LONS = np.array([0.0, 90.0, 0.0, 90.0])


def _field(grib_keys, name="x", **kwargs):
    """Build a 4-point field with ``field_from_grib_keys``."""
    return field_from_grib_keys(np.arange(4.0), grib_keys, name=name, **kwargs)


def _component_field(**components):
    """Build a 2-point field directly from earthkit-data component dicts."""
    return ekd.from_source("list-of-dicts", [{"data": {"values": np.zeros(2)}, **components}]).to_fieldlist()[0]


def _grib_field(sample, **keys):
    """Build a GRIB field in memory from an eccodes sample, with ``keys`` set on it."""
    import eccodes

    handle = eccodes.codes_grib_new_from_samples(sample)
    try:
        for key, value in keys.items():
            eccodes.codes_set(handle, key, value)
        message = eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)
    return ekd.from_source("memory", message).to_fieldlist()[0]


# --------------------------------------------------------------------------- #
# _levtype_to_level_type
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "levtype, level_type",
    [
        ("sfc", "surface"),
        ("pl", "pressure"),
        ("ml", "hybrid"),
        ("pt", "potential_temperature"),
        ("pv", "potential_vorticity"),
    ],
)
def test_levtype_to_level_type(levtype, level_type):
    """MARS abbreviations map to earthkit-data level type names, read from earthkit-data's registry."""
    assert _levtype_to_level_type()[levtype] == level_type


# --------------------------------------------------------------------------- #
# field_from_grib_keys
# --------------------------------------------------------------------------- #
def test_field_from_grib_keys_components():
    """Values, param, level, level type, valid datetime and geography land in their components."""
    field = _field(
        {"param": "t", "levelist": 850, "levtype": "pl"},
        name="t_850",
        valid_datetime=DATE,
        geography={"latitudes": LATS, "longitudes": LONS},
    )

    np.testing.assert_array_equal(field.to_numpy(), np.arange(4.0))
    assert field.get("parameter.variable") == "t"
    assert field.get("vertical.level") == 850
    assert field.get("vertical.level_type") == "pressure"
    assert field.get("vertical.abbreviation") == "pl"
    assert field.get("time.valid_datetime") == DATE
    lats, lons = field.geography.latlons(flatten=True)
    np.testing.assert_array_equal(lats, LATS)
    np.testing.assert_array_equal(lons, LONS)


def test_field_from_grib_keys_labels():
    """GRIB keys are also stored verbatim as labels, next to the name and the extra labels."""
    field = _field({"param": "t", "levelist": 850, "levtype": "pl"}, name="t_850", labels={"extra": 1})

    assert field.get("labels.name") == "t_850"
    assert field.get("labels.param") == "t"
    assert field.get("labels.levelist") == 850
    assert field.get("labels.levtype") == "pl"
    assert field.get("labels.extra") == 1


@pytest.mark.parametrize(
    "value",
    ["s", 1, 1.5, True, datetime.datetime(2020, 1, 1), datetime.timedelta(hours=6)],
)
def test_field_from_grib_keys_label_types_kept(value):
    """Every LABEL_TYPES value survives as a label."""
    assert _field({"param": "t", "key": value}).get("labels.key") == value


def test_field_from_grib_keys_non_label_types_dropped():
    """GRIB key values that are not LABEL_TYPES (e.g. lists) are not stored as labels."""
    field = _field({"param": "t", "pl": [1, 2]})
    assert field.get("labels.pl") is None


def test_field_from_grib_keys_name_wins_over_labels():
    """The ``name`` argument overrides a ``name`` passed in the extra labels."""
    assert _field({"param": "t"}, name="t", labels={"name": "other"}).get("labels.name") == "t"


def test_field_from_grib_keys_param_defaults_to_name():
    """Without a ``param`` GRIB key, the variable name is used as ``parameter.variable``."""
    field = _field({}, name="cos_latitude")
    assert field.get("parameter.variable") == "cos_latitude"


def test_field_from_grib_keys_level_zero_kept():
    """A falsy ``levelist`` (0) is still set as the vertical level."""
    assert _field({"param": "sd", "levelist": 0, "levtype": "sfc"}).get("vertical.level") == 0


def test_field_from_grib_keys_unknown_levtype():
    """An unknown levtype is kept as a label only and not registered as a new earthkit-data level type."""
    field = _field({"param": "lsm", "levtype": "o2d"})

    assert field.get("vertical.level_type") == "unknown"
    assert field.get("labels.levtype") == "o2d"
    assert "o2d" not in _levtype_to_level_type()


def test_field_from_grib_keys_without_optional_components():
    """Without levtype, levelist and valid datetime, those components are left unset."""
    field = _field({"param": "t"})

    assert field.get("vertical.level") is None
    assert field.get("vertical.level_type") == "unknown"
    assert field.get("time.valid_datetime") is None


@pytest.mark.parametrize(
    "grib_keys, level, is_pressure_level",
    [
        ({"param": "t", "levelist": 850, "levtype": "pl"}, 850, True),
        ({"param": "2t", "levtype": "sfc"}, None, False),
    ],
)
def test_field_from_grib_keys_variable_from_earthkit(grib_keys, level, is_pressure_level):
    """``Variable.from_earthkit`` finds the level and level type through the component API."""
    from anemoi.transform.variables import Variable

    variable = Variable.from_earthkit("x", _field(grib_keys))

    assert variable.level == level
    assert variable.is_pressure_level is is_pressure_level
    assert variable.grib_keys["levtype"] == grib_keys["levtype"]


# --------------------------------------------------------------------------- #
# FieldMetadata / get_metadata_dict
# --------------------------------------------------------------------------- #
def test_get_metadata_dict_returns_field_metadata():
    """``get_metadata_dict`` wraps the field in a ``FieldMetadata`` mapping."""
    assert isinstance(get_metadata_dict(_field({"param": "t"})), FieldMetadata)


def test_field_metadata_mars_keys_from_components():
    """MARS-style keys resolve from earthkit-data components, with levtype as the MARS abbreviation."""
    field = _component_field(
        parameter={"variable": "t"},
        vertical={"level": 500, "level_type": "pressure"},
        time={"valid_datetime": DATE},
    )
    md = get_metadata_dict(field)

    assert md["param"] == "t"
    assert md["levelist"] == 500
    assert md["levtype"] == "pl"
    assert md["valid_datetime"] == DATE


def test_field_metadata_iteration():
    """Iterating, ``len`` and ``dict`` cover only the common keys the field has."""
    md = get_metadata_dict(_field({"param": "t", "levelist": 850, "levtype": "pl", "grid": "O96"}, valid_datetime=DATE))

    assert dict(md) == {
        "param": "t",
        "levelist": 850,
        "levtype": "pl",
        "step": datetime.timedelta(0),
        "valid_datetime": DATE,
        "base_datetime": DATE,
    }
    assert len(md) == 6
    assert "grid" not in list(md)
    assert repr(md).startswith("FieldMetadata({")


def test_field_metadata_arbitrary_label_lookup():
    """Keys outside the common set can still be looked up explicitly, falling back to labels."""
    md = get_metadata_dict(_field({"param": "t", "grid": "O96"}))
    assert md["grid"] == "O96"


def test_field_metadata_falsy_values_kept():
    """Falsy values such as 0 are found, only None counts as missing."""
    md = get_metadata_dict(_field({"param": "sd", "levelist": 0, "number": 0}))

    assert md["levelist"] == 0
    assert md["number"] == 0


def test_field_metadata_missing_key():
    """A missing key raises KeyError on every lookup, and behaves as missing for ``in`` and ``get``."""
    md = get_metadata_dict(_field({"param": "t"}))

    for _ in range(2):
        with pytest.raises(KeyError):
            md["nope"]
    assert "nope" not in md
    assert md.get("nope", "default") == "default"


def test_field_metadata_levtype_unknown_is_missing():
    """A field without a level type has no ``levtype``, rather than ``"unknown"``."""
    md = get_metadata_dict(_field({"param": "t"}))

    assert "levtype" not in md
    assert md.get("levtype") is None


def test_field_metadata_levtype_from_label():
    """A levtype earthkit-data has no level type for falls back to the ``levtype`` label."""
    assert get_metadata_dict(_field({"param": "lsm", "levtype": "o2d"}))["levtype"] == "o2d"


def test_field_metadata_levtype_component_over_label():
    """The ``vertical`` component takes precedence over a ``levtype`` label."""
    field = _component_field(
        parameter={"variable": "t"},
        vertical={"level": 500, "level_type": "pressure"},
        labels={"levtype": "other"},
    )
    assert get_metadata_dict(field)["levtype"] == "pl"


@pytest.mark.parametrize(
    "sample, keys, levelist, levtype",
    [
        ("regular_ll_pl_grib2", {}, 1000, "pl"),
        ("regular_ll_sfc_grib2", {}, 0, "sfc"),
        # depth below sea: MARS levtype o2d, earthkit-data's abbreviation is "depthBelowSea"
        ("regular_ll_sfc_grib2", {"typeOfFirstFixedSurface": 160, "scaledValueOfFirstFixedSurface": 0}, 0, "o2d"),
    ],
)
def test_field_metadata_grib(sample, keys, levelist, levtype):
    """On GRIB fields, raw GRIB keys are found and the raw ``levtype`` wins over ``vertical.abbreviation``."""
    md = get_metadata_dict(_grib_field(sample, **keys))

    assert md["param"] == "t"
    assert md["levelist"] == levelist
    assert md["levtype"] == levtype
    assert md["shortName"] == "t"
    assert md["paramId"] == 130
    assert md["metadata.centre"] == "ecmf"


# --------------------------------------------------------------------------- #
# name_fields
# --------------------------------------------------------------------------- #
def test_name_fields():
    """The namer sees each field with a ``FieldMetadata`` and its result is set as ``labels.name``."""
    fields = ekd.create_fieldlist(
        [
            _field({"param": "t", "levelist": 850, "levtype": "pl"}, name="old", labels={"keep": "yes"}),
            _field({"param": "2t", "levtype": "sfc"}, name="old"),
        ]
    )
    seen = []

    def namer(field, md):
        seen.append(type(md))
        return "_".join(str(md[k]) for k in ("param", "levelist") if k in md)

    named = name_fields(fields, namer)

    assert seen == [FieldMetadata, FieldMetadata]
    assert [f.get("labels.name") for f in named] == ["t_850", "2t"]
    assert len(named.sel(**{"labels.name": "t_850"})) == 1
    # Other labels and components are kept, and the input is not modified.
    assert named[0].get("labels.keep") == "yes"
    assert named[0].get("parameter.variable") == "t"
    assert [f.get("labels.name") for f in fields] == ["old", "old"]


def test_name_fields_empty():
    """An empty fieldlist gives an empty fieldlist, without calling the namer."""

    def namer(field, md):
        raise AssertionError("namer called")

    assert len(name_fields(ekd.create_fieldlist([]), namer)) == 0
