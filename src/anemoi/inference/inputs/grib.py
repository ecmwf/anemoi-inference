# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.transform import FieldList

from .ekd import EkdInput

LOG = logging.getLogger(__name__)


class GribInput(EkdInput):
    """Handles GRIB input fields."""

    def set_private_attributes(self, state: Any, fields: FieldList) -> None:
        """Set private attributes for the state.

        Parameters
        ----------
        state : Any
            The state to set private attributes for.
        fields : FieldList
            The input fields.
        """
        # For now we just pass all the fields
        # Later, we can select a relevant subset (e.g. only one
        # level), to save memory

        # By sorting, we will have the most recent field last
        # no we can also use that list to write step 0
        super().set_private_attributes(state, fields)
        input_fields = fields.order_by("time.valid_datetime")

        state["_grib_templates_for_output"] = {field.name: field for field in input_fields}
