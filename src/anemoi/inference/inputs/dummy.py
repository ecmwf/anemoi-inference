# (C) Copyright 2024 Anemoi contributors.
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

import logging

import earthkit.data as ekd
import numpy as np

from anemoi.inference.testing import float_hash
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry
from .ekd import EkdInput

LOG = logging.getLogger(__name__)
SKIP_KEYS = ["date", "time", "step", "valid_datetime"]


@input_registry.register("dummy")
class DummyInput(EkdInput):
    """Dummy input used for testing."""

    trace_name = "dummy"

    def __init__(self, context, **kwargs) -> None:
        """Initialize the DummyInput.

        Parameters
        ----------
        context : Context
            The context for the input.
        """
        super().__init__(context, **kwargs)

    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        State
            The created input state.
        """
        assert date is not None, "date must be provided for dummy input"

        dates = [date + h for h in self.checkpoint.lagged]
        return self._create_input_state(self._fields(dates, self.variables), variables=None, date=date, **kwargs)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
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
            self._fields(dates, self.variables),
            dates=dates,
            current_state=current_state,
        )

    def _fields(self, dates: list[Date], variables) -> ekd.FieldList:
        """Generate fields for the given dates and variables.

        Parameters
        ----------
        dates : List[Date]
            List of dates for which to generate fields, by default None.
        variables : Optional[List[str]], optional
            List of variables for which to generate fields, by default None.

        Returns
        -------
        ekd.FieldList
            The generated fields.
        """

        LOG.info("Generating fields for %s", variables)

        typed_variables = self.checkpoint.typed_variables

        result = []
        for variable in variables:
            is_constant_in_time = typed_variables[variable].is_constant_in_time

            keys = {k: v for k, v in typed_variables[variable].grib_keys.items() if k not in SKIP_KEYS}

            for date in dates:
                x = float_hash(variable, dates[0] if is_constant_in_time else date)

                handle = dict(
                    values=np.ones(self.checkpoint.number_of_grid_points, dtype=np.float32) * x,
                    latitudes=np.zeros(self.checkpoint.number_of_grid_points, dtype=np.float32),
                    longitudes=np.zeros(self.checkpoint.number_of_grid_points, dtype=np.float32),
                    date=date.strftime("%Y%m%d"),
                    time=date.strftime("%H%M"),
                    name=variable,
                    **keys,
                )
                result.append(handle)

        return ekd.from_source("list-of-dicts", result)

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
