# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from .ekd import EkdInput

LOG = logging.getLogger(__name__)


class GribInput(EkdInput):
    """
    Handles GRIB input fields.
    """

    def set_private_attributes(self, state, input_fields):
        # For now we just pass all the fields
        # Later, we can select a relevant subset (e.g. only one
        # level), to save memory

        # By sorting, we will have the most recent field last
        # no we can also use that list to write step 0
        input_fields = input_fields.order_by("valid_datetime")

        state["_grib_templates_for_output"] = {field.metadata("name"): field for field in input_fields}
