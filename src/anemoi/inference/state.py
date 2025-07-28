# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import itertools

from anemoi.inference.types import State


def check_state(state: State, title: str = "<state>") -> None:
    """Check the state for consistency.

    Parameters
    ----------
    state : dict
        The state to check.
    title : str
        The title of the state (for logging).

    Raises
    ------
    ValueError
        If the state is not consistent.
    """

    if not isinstance(state, dict):
        raise ValueError(f"State '{title}' is not a dictionary: {state}")

    input = state.get("_input")

    if "fields" not in state:
        raise ValueError(f"State '{title}' does not contain 'fields': {state} ({input=})")

    shape = None

    for field, values in state["fields"].items():
        if shape is None:
            shape = values.shape
        elif shape != values.shape:
            raise ValueError(
                f"Field '{field}' in state '{title}' has different shape: " f"{shape} and {values.shape} ({input=})."
            )

    date = state.get("date")
    if date is None and len(state["fields"]) > 0:
        # date can be None for an empty input
        if not isinstance(date, datetime.datetime):
            raise ValueError(f"State '{title}' does not contain 'date', or it is not a datetime: {date} ({input=})")


def combine_states(*states: State) -> State:
    """Combine multiple states into one.

    Parameters
    ----------
    states : list
        The states to combine.

    Returns
    -------
    dict
        The combined state.
    """
    import numpy as np

    combined = states[0].copy()
    combined["fields"] = combined["fields"].copy()
    shape = None
    first_input = combined.get("_input")

    for state in states[1:]:

        this_input = state.get("_input")

        for name, values in itertools.chain(combined["fields"].items(), state.get("fields", {}).items()):
            if shape is None:
                shape = values.shape
            elif shape != values.shape:
                raise ValueError(
                    f"Field '{name}' has different shape in the states: "
                    f"{shape} and {values.shape}."
                    f" Input: {first_input} vs {this_input}."
                )

        if not set(combined["fields"]).isdisjoint(state["fields"]):
            raise ValueError(
                f"Some states have overlapping fields:"
                f" {set(combined['fields']).intersection(state['fields'])}"
                f" Input: {first_input} vs {this_input}."
            )

        combined["fields"].update(state.get("fields", {}))
        for key, value in state.items():
            if key == "fields":
                continue

            if key.startswith("_"):
                continue

            if combined.get(key) is None:
                combined[key] = value
                continue

            if value is None:
                continue

            if type(combined[key]) is not type(value):
                raise ValueError(
                    f"Key '{key}' has different types in the states: " f"{type(combined[key])} and {type(value)}."
                )

            if isinstance(value, np.ndarray) and isinstance(combined[key], np.ndarray):
                if not np.array_equal(combined[key], value):
                    raise ValueError(
                        f"Key '{key}' has different array values in the states: "
                        f"{combined[key]} and {value}."
                        f" Input: {first_input} vs {this_input}."
                    )
                continue

            if combined[key] != value:
                raise ValueError(
                    f"Key '{key}' has different values in the states: "
                    f"{combined[key]} and {value} ({shape})."
                    f" Input: {first_input} vs {this_input}."
                )

    return combined


def reduce_state(state: State) -> State:
    """Create a new state which is a projection of the original state on the last step in the multi-steps dimension.

    Parameters
    ----------
    state : State
        The original state.

    Returns
    -------
    State
        The reduced state.
    """
    reduced_state = state.copy()
    reduced_state["fields"] = {}
    for field, values in state["fields"].items():
        if len(values.shape) > 1:
            reduced_state["fields"][field] = values[-1, :]
        else:
            reduced_state["fields"][field] = values
    return reduced_state
