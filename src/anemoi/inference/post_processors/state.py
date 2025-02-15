# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property

from earthkit.data.core.fieldlist import Field
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.indexing.fieldlist import SimpleFieldList

LOG = logging.getLogger(__name__)


class StateFieldGeography:
    def __init__(self, field):
        self._field = field

    @property
    def shape(self):
        return self.field.shape


class StateFieldMetadata(RawMetadata):
    def __init__(self, field):
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

    def as_namespace(self, ns):
        assert ns == "mars"
        return {k: v for k, v in self.items() if k != "name"}

    @property
    def geography(self):
        return StateFieldGeography(self._field)


class StateField(Field):
    def __init__(self, name, values, state):
        self.name = name
        self.__values = values
        self.state = state

    def _values(self, dtype):
        return self.__values.astype(dtype)

    @property
    def shape(self):
        return self.__values.shape

    @cached_property
    def _metadata(self):
        return StateFieldMetadata(self)

    def __repr__(self):
        return f"{self.__class__.__name__ }({self._metadata})"


def wrap_state(state):
    """Transform a state dictionary into a field list."""
    assert isinstance(state["date"], datetime.datetime)  # Only works on single dates for now
    fields = [StateField(k, v, state) for k, v in state["fields"].items()]
    return SimpleFieldList(fields)


def unwrap_state(fields, state):
    """Transform a field list into a state dictionary."""
    new_fields = {}
    for n in fields:
        new_fields[n.metadata("name")] = n.to_numpy(flatten=True)
    state = state.copy()
    state["fields"] = new_fields
    return state
