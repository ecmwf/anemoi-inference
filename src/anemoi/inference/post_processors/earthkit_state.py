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

from anemoi.transform import Field
from anemoi.transform import FieldList

from anemoi.inference.inputs.ekd import _get_metadata_dict
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


def _create_state_field(name: str, values: FloatArray, state: State) -> Field:
    """Create an earthkit Field from a state field.

    Parameters
    ----------
    name : str
        The name of the field.
    values : FloatArray
        The values of the field.
    state : State
        The state information associated with the field.

    Returns
    -------
    Field
        The created field.
    """
    labels = {"name": name}
    # Add serialisable state entries as labels
    for k, v in state.items():
        if isinstance(v, (str, int, float, bool)):
            labels[k] = v

    return Field.from_components(
        values=values,
        parameter={"variable": name},
        labels=labels,
    )


# Keep StateField as a marker so unwrap_state can detect pass-through fields
class _StateFieldMarker:
    """Marker to identify fields created from state dictionaries."""

    pass


def wrap_state(state: State) -> FieldList:
    """Transform a state dictionary into an earthkit.data field list.

    Parameters
    ----------
    state : Dict[str, Any]
        The state dictionary to be transformed.

    Returns
    -------
    FieldList
        The transformed field list.
    """
    assert isinstance(state["date"], datetime.datetime)  # Only works on single dates for now
    fields = []
    for k, v in state["fields"].items():
        f = _create_state_field(k, v, state)
        # Tag the field so unwrap_state can detect it
        f._state_field_marker = True
        fields.append(f)
    return FieldList.from_fields(fields)


def unwrap_state(fields: FieldList, state: State, namer: Callable) -> State:
    """Transform a earthkit.data field list into a state dictionary.

    Parameters
    ----------
    fields : FieldList
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

    # namer(field: Field, metadata: Dict[str, Any]) -> str:

    for n in fields:
        md = _get_metadata_dict(n)
        name = namer(n, md)
        if getattr(n, "_state_field_marker", False):
            # StateField values are already flat 1D numpy arrays.
            # Use to_numpy() without flatten=True to avoid the always-copy
            # behavior of ndarray.flatten(). Combined with np.asarray in
            # _values(), this avoids unnecessary copies for pass-through
            # fields that were not transformed.
            new_fields[name] = n.to_numpy()
        else:
            new_fields[name] = n.to_numpy(flatten=True)

    state = state.copy()
    state["fields"] = new_fields

    return state
