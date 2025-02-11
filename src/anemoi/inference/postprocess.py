# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from earthkit.data import array_api

LOG = logging.getLogger(__name__)


class Noop:

    def __call__(self, source):
        yield from source


class Accumulator:
    """Accumulate fields from zero and return the accumulated fields"""

    def __init__(self, accumulations):
        self.accumulations = accumulations
        LOG.info("Accumulating fields %s", self.accumulations)

        self.accumulators = {}

    def __call__(self, source):
        for state in source:
            for accumulation in self.accumulations:
                if accumulation in state["fields"]:
                    an = array_api.get_backend(state["fields"][accumulation]).module

                    if accumulation not in self.accumulators:
                        self.accumulators[accumulation] = an.zeros_like(state["fields"][accumulation])

                    self.accumulators[accumulation] += an.maximum(
                        an.zeros((1,), device=state["fields"][accumulation].device), state["fields"][accumulation]
                    )
                    state["fields"][accumulation] = self.accumulators[accumulation]

            yield state
