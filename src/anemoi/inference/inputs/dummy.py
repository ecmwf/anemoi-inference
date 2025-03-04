# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any
from typing import List
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.grib.encoding import encode_message
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry
from .ekd import EkdInput

LOG = logging.getLogger(__name__)


@input_registry.register("dummy")
class DummyInput(EkdInput):
    """Dummy input used for testing."""

    trace_name = "dummy"

    def __init__(self, context: Context, *, namer: Optional[Any] = None, **kwargs: Any) -> None:
        """Initialize the DummyInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, namer=namer, **kwargs)

        self.templates = TemplateManager(self, "builtin")

    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The created input state.
        """
        if date is None:
            date = [datetime.datetime(2000, 1, 1)]

        dates = [date + h for h in self.checkpoint.lagged]
        return self._create_state(self._fields(dates), variables=None, date=date)

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            List of variables to load.
        dates : List[Date]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        Any
            The loaded forcings state.
        """
        return self._load_forcings_state(
            ekd.from_source("file", self.path),
            variables=variables,
            dates=dates,
            current_state=current_state,
        )

    def _fields(self, dates: Optional[List[Date]] = None, variables: Optional[List[str]] = None):
        """Generate fields for the given dates and variables.

        Parameters
        ----------
        dates : Optional[List[Date]], optional
            List of dates for which to generate fields, by default None.
        variables : Optional[List[str]], optional
            List of variables for which to generate fields, by default None.

        Returns
        -------
        SimpleFieldList
            The generated fields.
        """
        from earthkit.data.indexing.fieldlist import SimpleFieldList

        if variables is None:
            variables = self.checkpoint.variables_from_input(include_forcings=True)

        result = []
        for v in variables:
            template = self.templates.template(v, {})
            for date in dates:
                handle = encode_message(
                    values=np.zeros(self.checkpoint.number_of_grid_points, dtype=np.float32),
                    template=template,
                    metadata=dict(
                        date=date.strftime("%Y%m%d"),
                        time=date.strftime("%H%M"),
                        shortName=v,
                    ),
                )
                result.append(ekd.from_source("memory", handle.get_buffer())[0])

        for f in result:
            print(f)

        return SimpleFieldList(result)

    def template_lookup(self, name: str) -> dict:
        """Lookup a template by name.

        Parameters
        ----------
        name : str
            The name of the template to lookup.

        Returns
        -------
        dict
            The template dictionary.
        """
        # Unused, but required by the TemplateManager
        return {}
