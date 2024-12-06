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

import earthkit.data as ekd
import numpy as np
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from earthkit.data.indexing.fieldlist import FieldArray

from anemoi.inference.inputs.dataset import DatasetInput

LOG = logging.getLogger(__name__)


class Forcings(ABC):
    """Represents the forcings for the model."""

    def __init__(self, context):
        self.context = context
        self.checkpoint = context.checkpoint
        self.kinds = dict(unknown=True)  # Used for debugging

    @abstractmethod
    def load_forcings_array(self, dates, current_state) -> np.ndarray:
        """Load the forcings for the given dates."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def _state_to_numpy(self, state, variables, dates) -> np.ndarray:
        """Convert the state dictionary to a numpy array.
        This assumes that the state dictionary contains the fields for the given variables.
        And that the fields values are sorted by dates.
        """
        fields = state["fields"]
        result = np.stack([fields[v] for v in variables], axis=0)
        assert result.shape[:2] == (
            len(variables),
            len(dates),
        ), (result.shape, variables, dates)
        return result


class ComputedForcings(Forcings):
    """Compute forcings like `cos_julian_day` or `insolation`."""

    def __init__(self, context, variables, mask):
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(computed=True)  # Used for debugging

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, dates, current_state):

        LOG.debug("Adding dynamic forcings %s", self.variables)

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        source = UnstructuredGridFieldList.from_values(
            latitudes=current_state["latitudes"],
            longitudes=current_state["longitudes"],
        )

        ds = ekd.from_source("forcings", source, date=dates, param=self.variables)

        assert len(ds) == len(self.variables) * len(dates), (len(ds), len(self.variables), dates)

        def rename(f, _, metadata):
            return metadata["param"]

        ds = FieldArray([f.clone(name=rename) for f in ds])

        forcing = ds.order_by(name=self.variables, valid_datetime="ascending")

        # Forcing are sorted by `compute_forcings`  in the order (variable, date)

        return forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(self.variables), len(dates), -1)


class CoupledForcings(Forcings):
    """Retrieve forcings from the input."""

    def __init__(self, context, input, variables, mask):
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.input = input
        self.kinds = dict(retrieved=True)  # Used for debugging

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, dates, current_state):
        return self._state_to_numpy(
            self.input.load_forcings_state(variables=self.variables, dates=dates, current_state=current_state),
            self.variables,
            dates,
        )


class BoundaryForcings(Forcings):
    """Retrieve boundary forcings from the input."""

    def __init__(self, context, input, variables, variables_mask):
        super().__init__(context)
        self.variables = variables
        self.variables_mask = variables_mask
        assert isinstance(input, DatasetInput), "Currently only boundary forcings from dataset supported."
        self.input = input
        num_lam, num_other = input.ds.grids
        self.spatial_mask = np.array([False] * num_lam + [True] * num_other, dtype=bool)
        self.kinds = dict(retrieved=True)  # Used for debugging

    def __repr__(self):
        return f"{self.__class__.__name__}({self.variables})"

    def load_forcings_array(self, dates, current_state):
        data = self._state_to_numpy(
            self.input.load_forcings_state(variables=self.variables, dates=dates, current_state=current_state),
            self.variables,
            dates,
        )
        data = data[..., self.spatial_mask]

        expected_shape = (len(self.variables), len(dates), current_state["latitudes"][self.spatial_mask].size)
        assert data.shape == expected_shape, (data.shape, expected_shape)

        return data
