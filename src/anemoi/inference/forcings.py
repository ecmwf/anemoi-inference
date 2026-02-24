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
from datetime import timedelta
from typing import Any

import earthkit.data as ekd
import numpy as np
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from earthkit.data.indexing.fieldlist import FieldArray
from rich import print

from anemoi.inference.context import Context
from anemoi.inference.inputs.dataset import DatasetInput
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

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
    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        """Load the forcings for the given dates.

        Parameters
        ----------
        dates : List[Date]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the Forcings object."""
        return f"{self.__class__.__name__}"

    def _state_to_numpy(self, state: State, variables: list[str], dates: list[Date]) -> FloatArray:
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

    def __init__(self, context: Context, variables: list[str], mask: Any):
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

    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        """Load the computed forcings for the given dates.

        Parameters
        ----------
        dates : List[Date]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        LOG.debug("Adding dynamic forcings %s", self.variables)

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        source = UnstructuredGridFieldList.from_values(
            latitudes=current_state["latitudes"],
            longitudes=current_state["longitudes"],
        )

        ds = ekd.from_source("forcings", source, date=dates, param=self.variables)

        assert len(ds) == len(self.variables) * len(dates), (len(ds), len(self.variables), dates)

        def rename(field: ekd.Field, _: str, metadata: dict[str, Any]) -> str:
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

    def __init__(self, context: Context, input: Any, variables: list[str], mask: IntArray):
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

    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        """Load the forcings for the given dates.

        Parameters
        ----------
        dates : List[Any]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        return self._state_to_numpy(
            self.input.load_forcings_state(
                dates=dates,
                current_state=current_state,
            ),
            self.variables,
            dates,
        )


class PropagedForcings(CoupledForcings):
    initial_fields = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kinds["propagated"] = True  # Used for debugging
        self.current_variables = [var for var in self.variables if not var.endswith("_next")]

    def __repr__(self) -> str:
        return f"\033[36m{self.__class__.__name__}\033[0m({self.variables})"

    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        LOG.info(f"\033[32m ******* FORCINGS: Loading {self} for \033[33m dates {dates} \033[0m")
        new_dates = []
        ref_date = self.context.reference_date
        for date in dates:
            if date > ref_date:
                LOG.info(f"{date} > {ref_date}")
                if date.hour == ref_date.hour and date.minute == ref_date.minute and date.second == ref_date.second:
                    new_date = ref_date
                else:
                    new_date = ref_date - timedelta(days=1)
                new_date = new_date.replace(hour=date.hour, minute=date.minute, second=date.second)
                new_dates.append(new_date)
                LOG.info(f"\033[32m ******* FORCINGS: Rewriting from \033[31m{dates} to \033[33m{new_dates} \033[0m")
                continue

            new_dates.append(date)

        forcings_state = self.input.load_forcings_state(
            dates=new_dates,
            current_state=current_state,
        )

        if self.initial_fields is None:
            # take a copy just in case
            self.initial_fields = {var: np.copy(forcings_state["fields"][var]) for var in self.variables}

        if len(new_dates) == 1:
            _date = new_dates[0]
            timestep = self.context.checkpoint.timestep
            check = ref_date + timestep
            if _date.hour == check.hour and _date.minute == check.minute and _date.second == check.second:
                for var in self.current_variables:
                    next_var = var + "_next"
                    print(f"Copying {next_var} at ref date {ref_date} to {var} at {_date}")
                    forcings_state["fields"][var] = self.initial_fields[next_var][-1, :][np.newaxis, ...]

        array = self._state_to_numpy(
            forcings_state,
            self.variables,
            new_dates,
        )

        return array


class ConstantDateForcings(CoupledForcings):
    """Retrieve forcings from the first date in the input. Used, for example, in the interpolator where forcings are only available at the first time step of the input forecast."""

    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        """Load the forcings for the given dates.

        Parameters
        ----------
        dates : List[Any]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        constant_arr = self._state_to_numpy(
            self.input.load_forcings_state(
                dates=[dates[0]],
                current_state=current_state,
            ),
            self.variables,
            [dates[0]],
        )

        return np.concatenate([constant_arr for _ in range(len(dates))], axis=1)


class ConstantForcings(CoupledForcings):
    # Just to have a different __repr__
    pass


class BoundaryForcings(Forcings):
    """Retrieve boundary forcings from the input."""

    def __init__(self, context: Context, input: DatasetInput, variables: list[str], variables_mask: IntArray):
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

    def load_forcings_array(self, dates: list[Date], current_state: State) -> FloatArray:
        """Load the boundary forcings for the given dates.

        Parameters
        ----------
        dates : List[Date]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        FloatArray
            The loaded forcings as a numpy array.
        """
        data = self._state_to_numpy(
            self.input.load_forcings_state(dates=dates, current_state=current_state),
            self.variables,
            dates,
        )
        data = data[..., self.spatial_mask]

        expected_shape = (len(self.variables), len(dates), current_state["latitudes"][self.spatial_mask].size)
        assert data.shape == expected_shape, (data.shape, expected_shape)

        return data
