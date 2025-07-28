# (C) Copyright 2025 Anemoi contributors.
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
from typing import Optional

from anemoi.inference.context import Context
from anemoi.inference.state import combine_states
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..decorators import main_argument
from ..input import Input
from . import create_input
from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("split")
@main_argument("splits")
class SplitInput(Input):

    trace_name = "split input"

    def __init__(self, context: Context, *, splits: list, **kwargs: Any) -> None:

        all_variables = set(kwargs.get("variables"))
        assert all_variables is not None, "variables must be provided for split input"

        self.splits = {}

        fallback = None

        for s in splits:
            assert isinstance(s, dict), "each split must be a dictionary"
            assert "source" in s, "each split must have a 'source' key"
            assert "variables" in s, "each split must have a 'variables' key"

            vars = s["variables"]
            if (not vars) or (vars == "fallback"):
                fallback = s["source"]
                continue

            if not isinstance(vars, list):
                vars = [vars]

            vars = set(vars)

            if not set(vars) <= all_variables:
                raise ValueError(
                    f"Variables {vars} not in the provided variables {all_variables} ({set(vars) - all_variables})"
                )

            self.splits[tuple(vars)] = create_input(
                context,
                s["source"],
                variables=vars,
                purpose=s.get("purpose"),
            )

        for i, vars1 in enumerate(splits):
            vars1 = set(vars1)
            for j, vars2 in enumerate(splits):
                vars2 = set(vars2)
                if i == j:
                    continue
                if not vars1.isdisjoint(vars2):
                    raise ValueError(f"Splits {vars1} and {vars2} overlap in variables {vars1 & vars2}")

            all_variables -= vars1

        if all_variables:
            if fallback is None:
                raise ValueError(f"Variables {all_variables} not covered by any split and no fallback provided")

            LOG.debug(f"Variables {all_variables} not covered by any split, using fallback {fallback}")

            self.fallback = create_input(
                context,
                fallback,
                variables=sorted(all_variables),
                purpose=kwargs.get("purpose"),
            )

        self.splits = list(self.splits.values())

        super().__init__(context, **kwargs)

    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create the input state for the repeated-dates input.

        Parameters
        ----------
        date : Date or None
            The date for the input state.

        Returns
        -------
        State
            The created input state.
        """

        # TODO: Consider caching the result
        states = [split.create_input_state(date=date) for split in self.splits]

        state = combine_states(*states)
        state["_input"] = self
        state["date"] = date

        return state

    def load_forcings_state(self, *, dates: List[Date], current_state: State) -> State:
        """Load the forcings state for repeated dates input.

        Parameters
        ----------
        dates : list of Date
            The list of dates for which to repeat the fields.
        current_state : State
            The current state to use for loading.

        Returns
        -------
        State
            The loaded and repeated forcings state.
        """
        assert len(dates) > 0, "dates must not be empty for repeated dates input"

        states = [split.load_forcings_state(dates=dates, current_state=current_state) for split in self.splits]
        state = combine_states(*states)

        state["date"] = dates[-1]
        state["_input"] = self

        return state
