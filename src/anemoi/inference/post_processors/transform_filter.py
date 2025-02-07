# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


@post_processor_registry.register("transform_filter")
class TransformFilter(Processor):
    """Accumulate fields from zero and return the accumulated fields"""

    def __init__(self, context, accumulations=None):
        super().__init__(context)
        if accumulations is None:
            accumulations = context.checkpoint.accumulations

        self.accumulations = accumulations
        LOG.info("Accumulating fields %s", self.accumulations)

        self.accumulators = {}

    def process(self, state):
        for accumulation in self.accumulations:
            if accumulation in state["fields"]:
                if accumulation not in self.accumulators:
                    self.accumulators[accumulation] = np.zeros_like(state["fields"][accumulation])
                self.accumulators[accumulation] += np.maximum(0, state["fields"][accumulation])
                state["fields"][accumulation] = self.accumulators[accumulation]

        return state
