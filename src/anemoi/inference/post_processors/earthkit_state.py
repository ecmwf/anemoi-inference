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

import earthkit.data as ekd
from anemoi.transform.variables import Variable

from anemoi.inference.fields import LABEL_TYPES
from anemoi.inference.fields import field_from_grib_keys
from anemoi.inference.fields import get_metadata_dict
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


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

    # Scalar state entries travel with every field, as they did when the state was
    # wrapped in a flat metadata dictionary.
    state_labels = {k: v for k, v in state.items() if isinstance(v, LABEL_TYPES)}

    fields = [
        field_from_grib_keys(
            values,
            typed_variables[name].grib_keys,
            name=name,
            valid_datetime=state["date"],
            labels=state_labels,
        )
        for name, values in state["fields"].items()
    ]
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
        name = namer(n, get_metadata_dict(n))
        new_fields[name] = n.to_numpy(flatten=True)

    state = state.copy()
    state["fields"] = new_fields

    return state
