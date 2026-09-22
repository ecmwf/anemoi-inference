# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Code to wrap and unwrap state dictionaries into earthkit.data field lists. So that we can pass them through anemoi-transorm filters."""

import datetime
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import earthkit.data as ekd
from anemoi.transform.variables import Variable

from anemoi.inference.inputs.ekd import _get_metadata_dict
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)

# Types that can be stored verbatim as earthkit-data labels.
_LABEL_TYPES = (str, int, float, bool, datetime.datetime, datetime.timedelta)


@lru_cache(maxsize=1)
def _levtype_to_level_type() -> dict[str, str]:
    """Map MARS level type abbreviations to earthkit-data level type names.

    earthkit-data identifies level types by *name* (``"surface"``, ``"pressure"``,
    ...) and silently registers any unknown string as a brand new level type. MARS
    abbreviations (``"sfc"``, ``"pl"``, ...) must therefore be translated before
    being handed to ``Field.from_components``.

    Returns
    -------
    dict[str, str]
        Mapping of abbreviation to earthkit-data level type name.
    """
    from earthkit.data.field.component.level_type import LevelTypes

    return {t.value.abbreviation: t.value.name for t in LevelTypes}


def _create_state_field(name: str, values: FloatArray, state: State, variable: Variable) -> ekd.Field:
    """Create an earthkit-data field from one entry of a state dictionary.

    Parameters
    ----------
    name : str
        The name of the field.
    values : FloatArray
        The values of the field.
    state : State
        The state information associated with the field.
    variable : Variable
        The typed variable describing the field.

    Returns
    -------
    ekd.Field
        The created field.
    """
    grib_keys = variable.grib_keys

    # Everything is exposed verbatim as a label, so that `field.get("labels.<key>")`
    # returns exactly what the MARS-style metadata dict used to hold.
    labels: dict[str, Any] = {k: v for k, v in grib_keys.items() if isinstance(v, _LABEL_TYPES)}
    labels.update(name=name)
    labels.update({k: v for k, v in state.items() if isinstance(v, _LABEL_TYPES)})

    # The well-known keys are additionally set as proper components, so that
    # consumers using the earthkit-data component API (such as
    # `anemoi.transform.variables.Variable.from_earthkit`) can find them.
    vertical: dict[str, Any] = {}
    if (levelist := grib_keys.get("levelist")) is not None:
        vertical["level"] = levelist
    if (levtype := grib_keys.get("levtype")) is not None:
        # Unknown abbreviations are left out rather than registered as new level types.
        if (level_type := _levtype_to_level_type().get(levtype)) is not None:
            vertical["level_type"] = level_type

    return ekd.Field.from_components(
        values=values,
        parameter={"variable": grib_keys.get("param", name)},
        time={"valid_datetime": state["date"]},
        vertical=vertical or None,
        labels=labels,
    )


def wrap_state(state: State, typed_variables: dict[str, Variable]) -> ekd.FieldList:
    """Transform a state dictionary into an earthkit.data field list.

    Parameters
    ----------
    state : Dict[str, Any]
        The state dictionary to be transformed.
    typed_variables : dict[str, Variable]
        Metadata for the variables in the state.

    Returns
    -------
    ekd.FieldList
        The transformed field list.
    """
    assert isinstance(state["date"], datetime.datetime)  # Only works on single dates for now
    fields = [_create_state_field(name, values, state, typed_variables[name]) for name, values in state["fields"].items()]
    return ekd.create_fieldlist(fields)


def unwrap_state(fields: ekd.FieldList, state: State, namer: Callable) -> State:
    """Transform a earthkit.data field list into a state dictionary.

    Parameters
    ----------
    fields : ekd.FieldList
        The field list to be transformed.
    state : State
        The original state dictionary.
    namer : Callable
        A function to generate new field names.

    Returns
    -------
    Dict[str, Any]
        The transformed state dictionary.
    """
    new_fields = {}

    # namer(field: ekd.Field, metadata: Dict[str, Any]) -> str:

    for n in fields:
        name = namer(n, _get_metadata_dict(n))
        new_fields[name] = n.to_numpy(flatten=True)

    state = state.copy()
    state["fields"] = new_fields

    return state
