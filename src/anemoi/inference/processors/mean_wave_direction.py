# (C) Copyright 2024-2025 Anemoi contributors.
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
from . import processor_registry

LOG = logging.getLogger(__name__)


@processor_registry.register("mean_wave_direction")
class MeanWaveDirection(Processor):
    """Accumulate fields from zero and return the accumulated fields"""

    def __init__(self, context, *kwargs):
        super().__init__(context)

    def process(self, state):
        cos_mwd = state["fields"].pop("cos_mwd")
        sin_mwd = state["fields"].pop("sin_mwd")

        mwd = np.rad2deg(np.arctan2(sin_mwd, cos_mwd))
        mwd = np.where(mwd >= 360, mwd - 360, mwd)
        mwd = np.where(mwd < 0, mwd + 360, mwd)

        state["fields"]["mwd"] = mwd

        return state
