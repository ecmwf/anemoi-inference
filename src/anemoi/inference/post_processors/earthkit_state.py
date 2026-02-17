# (C) Copyright 2025 Anemoi contributors.
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
from functools import cached_property
from typing import Any
from typing import Callable
from typing import Dict

import earthkit.data as ekd
import numpy as np
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.indexing.fieldlist import SimpleFieldList

from anemoi.inference.types import FloatArray
from anemoi.inference.types import Shape
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


class StateFieldGeography:
    """Geographical information of a state field.

    Parameters
    ----------
    field : Any
        The field containing geographical data.
    """

    def __init__(self, field: Any) -> None:
        self._field = field

    @property
    def shape(self) -> Shape:
        """Tuple: Shape of the geographical field."""
        return self._field.shape


class StateFieldMetadata(RawMetadata):
    """Metadata for a state field.

    Parameters
    ----------
    field : Any
        The field containing metadata.
    """

    def __init__(self, field: Any) -> None:
        super().__init__(
            name=field.name,
            param=field.name,
            **{
                k: v
                for k, v in field.state.items()
                if isinstance(v, (str, int, float, bool, datetime.datetime, datetime.timedelta))
            },
        )
        self._field = field

    def as_namespace(self, ns: str) -> Dict[str, Any]:
        """Convert metadata to a specific namespace.

        Parameters
        ----------
        ns : str
            The namespace to convert the metadata to.

        Returns
        -------
        Dict[str, Any]
            The metadata in the specified namespace.
        """
        assert ns == "mars"
        return {k: v for k, v in self.items() if k != "name"}

    @property
    def geography(self) -> StateFieldGeography:
        """StateFieldGeography: Geographical information of the field."""
        return StateFieldGeography(self._field)


class StateField(ekd.Field):
    """State field containing name, values, and state information.

    Parameters
    ----------
    name : str
        The name of the field.
    values : FloatArray
        The values of the field.
    state : Dict[str, Any]
        The state information associated with the field.
    """

    def __init__(self, name: str, values: FloatArray, state: State) -> None:
        self.name = name
        self.__values = values
        self.state = state

    def _values(self, dtype: np.dtype) -> FloatArray:
        """Get the values of the field with a specific data type.

        Parameters
        ----------
        dtype : np.dtype
            The data type to convert the values to.

        Returns
        -------
        FloatArray
            The values of the field in the specified data type.
        """
        return self.__values.astype(dtype)

    @property
    def shape(self) -> Shape:
        """Tuple: Shape of the field."""
        return self.__values.shape

    @cached_property
    def _metadata(self) -> StateFieldMetadata:
        """StateFieldMetadata: Metadata of the field."""
        return StateFieldMetadata(self)

    def __repr__(self) -> str:
        """Str: String representation of the StateField."""
        return f"{self.__class__.__name__ }({self._metadata})"


def wrap_state(state: State) -> ekd.FieldList:
    """Transform a state dictionary into an earthkit.data field list.

    Parameters
    ----------
    state : Dict[str, Any]
        The state dictionary to be transformed.

    Returns
    -------
    ekd.FieldList
        The transformed field list.
    """
    assert isinstance(state["date"], datetime.datetime)  # Only works on single dates for now
    fields = [StateField(k, v, state) for k, v in state["fields"].items()]
    return SimpleFieldList(fields)


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
        name = namer(n, n.metadata())
        new_fields[name] = n.to_numpy(flatten=True)

    state = state.copy()
    state["fields"] = new_fields

    return state
