# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.transform.filters import filter_registry

from ..processor import Processor
from . import post_processor_registry
from .state import unwrap_state
from .state import wrap_state

LOG = logging.getLogger(__name__)


@post_processor_registry.register("mean_wave_direction")
class MeanWaveDirection(Processor):
    """Accumulate fields from zero and return the accumulated fields"""

    def __init__(self, context, **kwargs):
        super().__init__(context)
        self.filter = filter_registry.create("cos_sin_mean_wave_direction", **kwargs)

    def process(self, state):
        return unwrap_state(self.filter.backward(wrap_state(state)), state)
