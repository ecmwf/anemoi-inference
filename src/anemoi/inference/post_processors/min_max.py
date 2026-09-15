import logging

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


@post_processor_registry.register("min_max_clipper")
class MinMaxClipper(Processor):
    """Post process min and max predictions of the given field,
    by taking the min and max between the predicted field and predicted min/max."""

    def __init__(self, context: Context, max: str, min: str, target: str) -> None:
        super().__init__(context)
        self.max = max
        self.min = min
        self.target = target
        self.fields = (max, min, target)

    def process(self, state: State) -> State:
        fields = state["fields"]

        if (
            self.target not in fields
            or self.min not in fields
            or self.max not in fields
        ):
            LOG.warning(
                f"Found mismatch between inputs ({fields.keys()}) and filter metadata {self.fields}"
            )
            return state

        fields[self.max] = np.maximum(fields[self.target], fields[self.max])
        fields[self.min] = np.minimum(fields[self.target], fields[self.min])
        state["fields"] = fields
        return state

    def __repr__(self) -> str:
        """Return a string representation of the processor.

        Returns
        -------
        str
            The class name of the processor.
        """
        return f"MinMaxFixer(min={self.min}, max={self.max}, target={self.target})"
