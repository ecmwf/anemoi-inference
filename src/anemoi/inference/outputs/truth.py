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

from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import ForwardOutput
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("truth")
@main_argument("output")
class TruthOutput(ForwardOutput):
    """Write the input state at the same time for each output state.

    Can only be used for inputs with that have access to the time of
    the forecasts, effectively only for times in the past.
    """

    def __init__(self, context: DefaultRunner, output, **kwargs: Any) -> None:
        """Initialise the TruthOutput.

        Parameters
        ----------
        context : Context
            The context for the output.
        output :
            The output configuration.
        kwargs : dict
            Additional keyword arguments.
        """
        if not isinstance(context, DefaultRunner):
            raise ValueError("TruthOutput can only be used with `DefaultRunner`")

        super().__init__(context, output, None, **kwargs)
        self._input = context.create_prognostics_input()

    def modify_state(self, state: State) -> State:
        """Modify state by overriding it with the truth state."""
        truth_state = self.reduce(self._input.create_input_state(date=state["date"]))
        truth_state["step"] = state.get("step", 0)
        return truth_state

    def __repr__(self) -> str:
        """Return a string representation of the TruthOutput.

        Returns
        -------
        str
            String representation of the TruthOutput.
        """
        return f"TruthOutput({self.output})"
