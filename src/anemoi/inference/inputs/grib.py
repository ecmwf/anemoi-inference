# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc
import datetime
import logging
from typing import Any
from typing import Dict
from typing import List

import earthkit.data as ekd
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.readers import Reader

from .ekd import EarthKitInput

LOG = logging.getLogger(__name__)


class GribInput(EarthKitInput):
    """Handles grib files."""

    def _raw_state_fieldlist(self, dates: List[datetime.datetime], variables: List[str]) -> ekd.FieldList:
        """Load the raw state fieldlist for the given dates and variables."""

        # NOTE: this is deferred because it's only passed as a callable
        def _name(field: Any, _: Any, original_metadata: Dict[str, Any]) -> str:
            return self._namer(field, original_metadata)

        data = self._earthkit_reader(dates, variables)
        data = FieldArray([f.clone(name=_name) for f in data])

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        # it's actually executed here
        data = data.sel(name=variables, valid_datetime=valid_datetime).order_by(
            name=variables, valid_datetime="ascending"
        )

        return data

    @abc.abstractmethod
    def _earthkit_reader(self, dates: datetime.datetime, variables: List[str]) -> Reader:
        """Return the earthkit reader for the input."""
        pass

    def set_private_attributes(self, state: Any, input_fields: ekd.FieldList) -> None:
        """Set private attributes for the state.

        Parameters
        ----------
        state : Any
            The state to set private attributes for.
        input_fields : ekd.FieldList
            The input fields.
        """
        # For now we just pass all the fields
        # Later, we can select a relevant subset (e.g. only one
        # level), to save memory

        # By sorting, we will have the most recent field last
        # no we can also use that list to write step 0
        input_fields = input_fields.order_by("valid_datetime")

        state["_grib_templates_for_output"] = {field.metadata("name"): field for field in input_fields}
