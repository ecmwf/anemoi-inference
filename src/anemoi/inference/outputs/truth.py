# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from anemoi.inference.config import Configuration
from anemoi.inference.types import State

from ..context import Context
from ..output import ForwardOutput
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("truth")
class TruthOutput(ForwardOutput):
    """Write the input state at the same time for each output state.

    Can only be used for inputs with that have access to the time of
    the forecasts, effectively only for times in the past.
    """

    def __init__(self, context: Context, output: Configuration, **kwargs: Any) -> None:
        """Initialize the TruthOutput.

        Parameters
        ----------
        context : Context
            The context for the output.
        output : Configuration
            The output configuration.
        kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, output, **kwargs)
        self._input = self.context.create_input()

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        truth_state = self._input.create_input_state(date=state["date"])
        reduced_state = self.reduce(truth_state)
        self.output.write_state(reduced_state)

    def __repr__(self) -> str:
        """Return a string representation of the TruthOutput.

        Returns
        -------
        str
            String representation of the TruthOutput.
        """
        return f"TruthOutput({self.output})"
