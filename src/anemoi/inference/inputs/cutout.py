# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import List
from typing import Optional

import numpy as np

from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("cutout")
class Cutout(Input):
    """Combines one or more LAMs into a global source using cutouts."""

    def __init__(self, context, **sources: dict[str, dict]):
        """Create a cutout input from a list of sources.

        Parameters
        ----------
        context : dict
            The context runner.
        sources : dict of sources
            A dictionary of sources to combine.
        """
        super().__init__(context)

        self.sources: dict[str, Input] = {}
        self.masks: dict[str, np.ndarray] = {}
        for src, cfg in sources.items():
            mask = cfg.pop("mask", f"{src}/cutout_mask")
            self.sources[src] = create_input(context, cfg)
            self.masks[src] = self.sources[src].checkpoint.load_supporting_array(mask)

    def __repr__(self):
        return f"Cutout({self.sources})"

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

        LOG.info(f"Concatenating states from {self.sources}")
        sources = list(self.sources.keys())

        state = self.sources[sources[0]].create_input_state(date=date)
        for source in sources[1:]:
            mask = self.masks[source]
            _state = self.sources[source].create_input_state(date=date)

            state["latitudes"] = np.concatenate([state["latitudes"], _state["latitudes"][..., mask]], axis=-1)
            state["longitudes"] = np.concatenate([state["longitudes"], _state["longitudes"][..., mask]], axis=-1)
            for field, values in state["fields"].items():
                state["fields"][field] = np.concatenate([values, _state["fields"][field][..., mask]], axis=-1)

        return state

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
        State
            The loaded forcings state.
        """

        sources = list(self.sources.keys())
        fields = self.sources[sources[0]].load_forcings_state(
            variables=variables, dates=dates, current_state=current_state
        )["fields"]
        for source in sources[1:]:
            mask = self.masks[source]
            _fields = self.sources[source].load_forcings_state(
                variables=variables, dates=dates, current_state=current_state
            )["fields"]
            for field in fields:
                fields[field] = np.concatenate([fields[field], _fields[field][..., mask]], axis=-1)

        current_state["fields"] |= fields
        return current_state
