# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from abc import abstractmethod

from . import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """
    Handles grib
    """

    def __init__(self, checkpoint, *, verbose=True):
        super().__init__(checkpoint, verbose=verbose)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables

    def write_initial_state(self, state):
        pass

    def write_state(self, state):
        reference_date = state["reference_date"]
        date = state["date"]

        for name, value in state["fields"].items():
            variable = self.typed_variables[name]
            if variable.is_accumulation:
                continue

            keys = variable.grib_keys
            keys.update(
                type="fc",
                date=reference_date.strftime("%Y-%m-%d"),
                time=reference_date.strftime("%H%M"),
                step=(date - reference_date).total_seconds() // 3600,
                expver="anem",
            )

            keys["class"] = "ml"

            self.write_message(value, **keys)

    @abstractmethod
    def write_message(self, message, *args, **kwargs):
        pass
