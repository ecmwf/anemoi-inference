# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping

import numpy as np

from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.inputs import input_registry
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


def contains_key(obj, key: str) -> bool:
    """Recursively check if `key` exists anywhere in a nested config (dict/DotDict/lists)."""
    if isinstance(obj, Mapping):
        if key in obj:
            return True
        return any(contains_key(v, key) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(contains_key(v, key) for v in obj)
    return False


def _mask_and_combine_states(
    existing_state: State,
    new_state: State,
    mask: np.ndarray | slice,
    fields: Iterable[str],
) -> State:
    """Mask and combine two states.

    Existing can be None, in which case new_state masked is returned.

    Parameters
    ----------
    existing_state : State
        The state to be combined into.
    new_state : State
        The other state to combine.
    mask : np.ndarray
        The mask to apply to new_state.
    fields: Iterable[str]
        The fields to combine in the states

    Returns
    -------
    State
        The combined state
    """
    was_empty = existing_state == {}

    for field in fields:
        if was_empty:
            existing_state[field] = new_state[field][..., mask]
        else:
            existing_state[field] = np.concatenate(
                [existing_state[field], new_state[field][..., mask]],
                axis=-1,
            )

    return existing_state


def _realise_mask(sli: slice | np.ndarray, state: dict) -> np.ndarray:
    """Realise a slice or array into an bool array based on the state shape."""
    return np.ones(len(state["latitudes"]), dtype=bool)[sli]


def _extract_and_add_private_attributes(
    private_attributes: defaultdict[str, dict], state: State, name: str
) -> defaultdict[str, dict]:
    """Extract and add private attributes to the state.

    Will nest the attributes under the name provided, maintaining the original key.

    Parameters
    ----------
    private_attributes : defaultdict[str, dict]
        The dictionary to contain private attributes.
    state : State
        The state to which the attributes will be retrieved.
    name : str
        The name of the sub-state to record the attributes under.

    Returns
    -------
    defaultdict[str, dict]
        The updated private attributes dictionary.
    """

    for key in (k for k in state.keys() if k.startswith("_")):
        private_attributes[key][name] = state[key]
    return private_attributes


@input_registry.register("cutout")
class Cutout(Input):
    """Combines one or more LAMs into a global source using cutouts."""

    # TODO: Does this need an ordering?

    def __init__(
        self,
        context,
        *,
        variables: list[str] | None = None,
        pre_processors: list[ProcessorConfig] | None = None,
        purpose: str | None = None,
        **sources: dict[str, dict],
    ):
        """Create a cutout input from a list of sources.

        Parameters
        ----------
        context : dict
            The context runner.
        sources : dict of sources
            A dictionary of sources to combine.
        variables : list[str] | None
            List of variables to be handled by the input, or None for a sensible default variables.
        pre_processors : Optional[List[ProcessorConfig]], default None
            Pre-processors to apply to the input. Note that pre_processors are applied to each sub-input.
        purpose : Optional[str]
            The purpose of the input.
        """

        super().__init__(context, variables=variables, pre_processors=pre_processors, purpose=purpose)

        self.sources: dict[str, Input] = {}
        self.masks: dict[str, np.ndarray | slice] = {}

        for src, cfg in sources.items():
            if isinstance(cfg, str):
                mask = f"{src}/cutout_mask"
            else:
                cfg = cfg.copy()
                mask = cfg.pop("mask", f"{src}/cutout_mask")

            if contains_key(cfg, "pre_processors"):
                combined_pre_processors = (pre_processors or []).extend(cfg.get("pre_processors", []))
                self.sources[src] = create_input(
                    context, cfg, variables=variables, pre_processors=combined_pre_processors, purpose=purpose
                )
            else:
                self.sources[src] = create_input(
                    context,
                    cfg,
                    variables=variables,
                    purpose=purpose,
                    pre_processors=pre_processors,
                )

            if isinstance(mask, str):
                self.masks[src] = self.sources[src].checkpoint.load_supporting_array(mask)
            else:
                self.masks[src] = mask if mask is not None else slice(0, None)  # type: ignore

    def __repr__(self):
        """Return a string representation of the Cutout object."""
        return f"Cutout({self.sources})"

    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        **kwargs : dict
            Additional keyword arguments for the source input state creation.

        Returns
        -------
        State
            The created input state.
        """

        LOG.info(f"Concatenating states from {self.sources}")

        _mask_private_attributes = {}
        _private_attributes = defaultdict(dict)

        combined_state = {}

        for source in self.sources.keys():
            source_state = self.sources[source].create_input_state(date=date, **kwargs)
            source_mask = self.masks[source]

            # Create the mask front padded with zeros
            # to match the length of the combined state
            _realised_mask = _realise_mask(source_mask, source_state)
            current_length = len(combined_state.get("latitudes", []))

            _mask_private_attributes[source] = np.concatenate(
                (
                    np.zeros((current_length,), dtype=bool),
                    _realised_mask,
                ),
                axis=-1,
            )
            # Combine the private attributes
            _private_attributes = _extract_and_add_private_attributes(_private_attributes, source_state, source)

            combined_state = _mask_and_combine_states(
                combined_state, source_state, source_mask, ["longitudes", "latitudes"]
            )
            combined_state["fields"] = _mask_and_combine_states(
                combined_state.get("fields", {}), source_state["fields"], source_mask, source_state["fields"].keys()
            )

            for key in (k for k in source_state.keys() if not k.startswith("_")):
                combined_state.setdefault(key, source_state[key])

        # Pad the masks to the total length of the combined state
        # then add them to the private attributes
        total_length = len(combined_state["latitudes"])
        for sub_mask in _mask_private_attributes:
            mask = _mask_private_attributes[sub_mask]
            _mask_private_attributes[sub_mask] = np.pad(mask, (0, total_length - len(mask)), constant_values=False)

        _private_attributes["_mask"] = _mask_private_attributes

        combined_state.update(_private_attributes)
        return combined_state

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
        State
            The loaded forcings state.
        """

        combined_fields = {}

        for source in self.sources.keys():
            source_state = self.sources[source].load_forcings_state(dates=dates, current_state=current_state)["fields"]
            source_mask = self.masks[source]

            combined_fields = _mask_and_combine_states(combined_fields, source_state, source_mask, source_state.keys())

        current_state["fields"] |= combined_fields
        return current_state
