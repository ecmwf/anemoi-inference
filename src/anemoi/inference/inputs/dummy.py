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
from typing import Any
from typing import List
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.testing import float_hash
from anemoi.inference.types import Date

from . import input_registry
from .ekd import EarthKitInput

LOG = logging.getLogger(__name__)
SKIP_KEYS = ["date", "time", "step"]


@input_registry.register("dummy")
class DummyInput(EarthKitInput):
    """Dummy input used for testing."""

    trace_name = "dummy"

    def __init__(self, context: Context, *, namer: Optional[Any] = None, **kwargs: Any) -> None:
        """Initialize the DummyInput.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, namer=namer, **kwargs)

    def _raw_state_fieldlist(self, dates: List[Date], variables: Optional[List[str]] = None) -> ekd.FieldList:
        """Generate fields for the given dates and variables.

        Parameters
        ----------
        dates : List[Date]
            List of dates for which to generate fields, by default None.
        variables : Optional[List[str]], optional
            List of variables for which to generate fields, by default None.

        Returns
        -------
        Reader
            The generated fields.
        """

        if variables is None:
            variables = self.checkpoint.variables_from_input(include_forcings=True)

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
