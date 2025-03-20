# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

import earthkit.data as ekd
import numpy as np
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from earthkit.data.indexing.fieldlist import FieldArray

from anemoi.inference.context import Context
from anemoi.inference.input import Input
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from .inputs.dataset import DatasetInput

LOG = logging.getLogger(__name__)


class Forcings(ABC):
    """Represents the forcings for the model."""

    def __init__(self, context: Context):
        """Initialize the Forcings object.

        Parameters
        ----------
        context : Context
            The context for the forcings.
        """
        self.context = context
        self.checkpoint = context.checkpoint
        self.kinds = dict(unknown=True)  # Used for debugging

    @abstractmethod
    def load_forcings_array(self, date: Date, *, initial: bool) -> FloatArray:
        """Load the forcings for the given dates.

        Parameters
        ----------
        date : Date
            The date for which to load the forcings.
        initial : bool
            Whether the forcings are for the initial conditions.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the Forcings object."""
        return f"{self.__class__.__name__}"

    def _state_to_numpy(self, state: State, variables: List[str], dates: List[Date]) -> FloatArray:
        """Convert the state dictionary to a numpy array.
        This assumes that the state dictionary contains the fields for the given variables.
        And that the fields values are sorted by dates.

        Parameters
        ----------
        state : State
            The state dictionary.
        variables : List[str]
            The list of variables.
        dates : List[Date]
            The list of dates.

        Returns
        -------
        FloatArray
            The state as a numpy array.
        """
        fields = state["fields"]

        if len(dates) == 1:
            result = np.stack([fields[v] for v in variables], axis=0)
            if len(result.shape) == 2:
                result = result[:, np.newaxis]
        else:
            result = np.stack([fields[v] for v in variables], axis=0)

        # Assert that the shape is correct: (variables, dates, values)

        assert len(result.shape) == 3 and result.shape[0] == len(variables) and result.shape[1] == len(dates), (
            result.shape,
            variables,
            dates,
        )

        return result


class ComputedForcings(Forcings):
    """Compute forcings like `cos_julian_day` or `insolation`."""

    trace_name = "computed"

    def __init__(self, context: Context, variables: List[str], mask: Any):
        """Initialize the ComputedForcings object.

        Parameters
        ----------
        context : Context
            The context for the forcings.
        variables : List[str]
            The list of variables to compute.
        mask : Any
            The mask to apply to the forcings.
        """
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(computed=True)  # Used for debugging

    def __repr__(self) -> str:
        """Return a string representation of the ComputedForcings object."""
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, date: Date, *, initial: bool) -> FloatArray:

        LOG.debug("Adding dynamic forcings %s", self.variables)

        dates = [date + h for h in self.checkpoint.lagged] if initial else [date]
        source = UnstructuredGridFieldList.from_values(
            latitudes=self.checkpoint.latitudes,
            longitudes=self.checkpoint.longitudes,
        )

        ds = ekd.from_source("forcings", source, date=dates, param=self.variables)

        assert len(ds) == len(self.variables) * len(dates), (len(ds), len(self.variables), dates)

        def rename(field: ekd.Field, _: str, metadata: Dict[str, Any]) -> str:
            return metadata["param"]

        ds = FieldArray([f.clone(name=rename) for f in ds])

        forcing = ds.order_by(name=self.variables, valid_datetime="ascending")

        # Forcing are sorted by `compute_forcings`  in the order (variable, date)

        return forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(self.variables), len(dates), -1)


class CoupledForcings(Forcings):
    """Retrieve forcings from the input."""

    @property
    def trace_name(self) -> str:
        """Return the trace name of the input."""
        return self.input.trace_name

    def __init__(self, context: Context, input: Input, variables: List[str], mask: IntArray):
        """Initialize the CoupledForcings object.

        Parameters
        ----------
        context : Context
            The context for the forcings.
        input : Any
            The input object.
        variables : List[str]
            The list of variables to retrieve.
        mask : IntArray
            The mask to apply to the forcings.
        """
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.input = input
        self.kinds = dict(retrieved=True)  # Used for debugging

    def __repr__(self) -> str:
        """Return a string representation of the CoupledForcings object."""
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, date: Date, initial=True) -> FloatArray:

        return self._state_to_numpy(
            self.input.load_forcings_state(
                date=date,
                variables=self.variables,
                initial=initial,
            ),
            self.variables,
            [date],
        )


class BoundaryForcings(Forcings):
    """Retrieve boundary forcings from the input."""

    def __init__(self, context: Context, input: DatasetInput, variables: List[str], variables_mask: IntArray):
        """Initialize the BoundaryForcings object.

        Parameters
        ----------
        context : Context
            The context for the forcings.
        input : DatasetInput
            The input object.
        variables : List[str]
            The list of variables to retrieve.
        variables_mask : IntArray
            The mask to apply to the forcings.
        """
        super().__init__(context)
        self.variables = variables
        self.variables_mask = variables_mask
        assert isinstance(input, DatasetInput), "Currently only boundary forcings from dataset supported."
        self.input = input
        if "output_mask" in context.checkpoint._supporting_arrays:
            self.spatial_mask = ~context.checkpoint.load_supporting_array("output_mask")
        else:
            self.spatial_mask = np.array([False] * len(input["latitudes"]), dtype=bool)
        self.kinds = dict(retrieved=True)  # Used for debugging

    def __repr__(self) -> str:
        """Return a string representation of the BoundaryForcings object."""
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, date: Date, initial: bool) -> FloatArray:

        data = self._state_to_numpy(
            self.input.load_forcings_state(date=date, variables=self.variables, initial=initial),
            self.variables,
            [date],
        )
        data = data[..., self.spatial_mask]

        expected_shape = (len(self.variables), 1, self.checkpoint.latitudes[self.spatial_mask].size)
        assert data.shape == expected_shape, (data.shape, expected_shape)

        return data
