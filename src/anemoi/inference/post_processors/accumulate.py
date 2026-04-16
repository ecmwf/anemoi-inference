# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from datetime import timedelta

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


@post_processor_registry.register("accumulate_from_start_of_forecast")
class Accumulate(Processor):
    """Accumulate fields from zero and return the accumulated fields.

    Parameters
    ----------
    context : Any
        The context in which the processor is running.
    metadata : Metadata
        Metadata corresponding to the dataset this processor is handling.
    accumulations : Optional[List[str]], optional
        List of fields to accumulate, by default None.
        If None, the fields are taken from the context's checkpoint.
    allow_negative : bool, optional
        Whether to allow negative values in the accumulation, by default False.
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        accumulations: list[str] | None = None,
        allow_negative: bool = False,
    ) -> None:
        super().__init__(context, metadata)
        if accumulations is None:
            accumulations = metadata.accumulations

        self.accumulations = accumulations
        self.allow_negative = allow_negative
        LOG.info("Accumulating fields %s (allow_negative=%s)", self.accumulations, self.allow_negative)

        self.accumulators: dict[str, FloatArray] = {}
        self.step_zero = timedelta(0)

    def process(self, state: State) -> State:
        """Process the state to accumulate specified fields.

        Parameters
        ----------
        state : State
            The state containing fields to be accumulated.

        Returns
        -------
        State
            The updated state with accumulated fields.
        """
        state = state.copy()
        state.setdefault("start_steps", {})
        for accumulation in self.accumulations:
            if accumulation in state["fields"]:
                if accumulation not in self.accumulators:
                    self.accumulators[accumulation] = np.zeros_like(state["fields"][accumulation])
                value = state["fields"][accumulation]
                if not self.allow_negative:
                    value = np.maximum(0, value)
                self.accumulators[accumulation] += value
                state["fields"][accumulation] = self.accumulators[accumulation]
                state["start_steps"][accumulation] = self.step_zero

        return state

    def __repr__(self) -> str:
        """Return a string representation of the Accumulate object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"Accumulate({self.accumulations})"
