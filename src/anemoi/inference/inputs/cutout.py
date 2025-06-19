# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.types import Date
from anemoi.inference.types import State

from . import input_registry

LOG = logging.getLogger(__name__)


def _mask_and_combine_states(
    combined_state: State,
    new_state: State,
    combined_mask: Union[np.ndarray, slice],
    mask: np.ndarray,
    fields: Iterable[str],
) -> State:
    """Mask and combine two states.

    Parameters
    ----------
    combined_state : State
        The state to be combined into.
    new_state : State
        The other state to combine.
    combined_mask : Optional[np.ndarray]
        The mask to apply to combined_state. If None, no mask is applied
    mask : np.ndarray
        The mask to apply to new_state.
    fields: Iterable[str]
        The fields to combine in the states

    Returns
    -------
    State
        The combined state
    """
    for field in fields:
        combined_state[field] = np.concatenate(
            [combined_state[field][..., combined_mask], new_state[field][..., mask]],
            axis=-1,
        )

    return combined_state


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
            if isinstance(cfg, str):
                mask = f"{src}/cutout_mask"
            else:
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

        combined_state = self.sources[sources[0]].create_input_state(date=date)
        combined_mask = self.masks[sources[0]]
        for source in sources[1:]:
            mask = self.masks[source]
            new_state = self.sources[source].create_input_state(date=date)

            combined_state = _mask_and_combine_states(
                combined_state, new_state, combined_mask, mask, ["longitudes", "latitudes"]
            )
            combined_state["fields"] = _mask_and_combine_states(
                combined_state["fields"], new_state["fields"], combined_mask, mask, combined_state["fields"]
            )
            combined_mask = slice(0, None)

        return combined_state

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
        combined_fields = self.sources[sources[0]].load_forcings_state(
            variables=variables, dates=dates, current_state=current_state
        )["fields"]
        combined_mask = self.masks[sources[0]]
        for source in sources[1:]:
            mask = self.masks[source]
            new_fields = self.sources[source].load_forcings_state(
                variables=variables, dates=dates, current_state=current_state
            )["fields"]
            combined_fields = _mask_and_combine_states(
                combined_fields, new_fields, combined_mask, mask, combined_fields
            )
            combined_mask = slice(0, None)

        current_state["fields"] |= combined_fields
        return current_state
