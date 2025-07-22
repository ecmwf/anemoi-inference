# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Dummy input used for testing.

It will generate fields with constant values for each variable and date.
These values are then tested in the mock model.
"""

import datetime
import logging
from typing import List
from typing import Optional

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)
SKIP_KEYS = ["date", "time", "step"]


@input_registry.register("empty")
class EmptyInput(Input):

    trace_name = "empty"

    def __init__(
        self,
        context: Context,
        *,
        variables: Optional[List[str]],
        pre_processors: Optional[List[ProcessorConfig]] = None,
    ) -> None:

        super().__init__(context, variables=variables, pre_processors=pre_processors)
        assert variables in (None, []), "EmptyInput should not have variables"

    def create_input_state(self, *, date: Optional[Date]) -> State:

        if date is None:
            date = datetime.datetime(2000, 1, 1)

        dates = [date + h for h in self.checkpoint.lagged]
        return dict(date=dates, latitudes=None, longitudes=None, fields=dict())

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        return dict(date=dates, latitudes=None, longitudes=None, fields=dict())
