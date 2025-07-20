# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import List
from typing import Optional

from anemoi.utils.dates import as_datetime

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..input import Input
from . import create_input
from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("repeated-dates")
class GribFileInput(Input):
    """Handles constants."""

    trace_name = "repeated dates"

    def __init__(
        self,
        context: Context,
        *,
        variables: Optional[List[str]],
        pre_processors: Optional[List[ProcessorConfig]] = None,
        source: str,
        mode: str = "constant",  # Same as "anemoi-dataset"
        **kwargs: Any,
    ) -> None:

        self.date = kwargs.pop("date", None)
        assert self.date is not None, "date must be provided for repeated-dates input"

        self.date = as_datetime(self.date)
        self.source = create_input(context, source, variables=variables)
        self.mode = mode

        assert self.mode in ["constant"], f"Unknown mode {self.mode}"

        super().__init__(context, variables=variables, pre_processors=pre_processors, **kwargs)

    def create_input_state(self, *, date: Optional[Date]) -> State:
        return self.source.create_input_state(date=self.date)

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        assert len(dates) > 0, "dates must not be empty for repeated dates input"

        state = self.source.load_forcings_state(
            variables=variables,
            dates=[self.date],
            current_state=current_state,
        )

        fields = state["fields"]

        for name, data in fields.items():
            assert len(data.shape) == 2, data.shape
            assert data.shape[0] == 1, data.shape
            fields[name] = data.repeat(len(dates), axis=0)

        return state
