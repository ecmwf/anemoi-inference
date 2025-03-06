# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)

# This is because forcings are assumed to already be in the
# state dictionary, so we don't need to load them from anywhere.


class NoForcings(Forcings):
    """No forcings."""

    def __init__(self, context: Any, variables: List[str], mask: IntArray) -> None:
        """Initialize the NoForcings.

        Parameters
        ----------
        context : object
            The context for the forcings.
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.
        """
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(unknown=True)

    def load_forcings_state(self, state: State, date: str) -> None:
        """Load forcings state.

        Parameters
        ----------
        state : State
            The state to load the forcings into.
        date : str
            The date for which to load the forcings.
        """
        pass


@runner_registry.register("simple")
class SimpleRunner(Runner):
    """Use that runner when using the low level API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SimpleRunner.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def create_constant_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created dynamic computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def create_constant_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created constant coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return []

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created dynamic coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return []
