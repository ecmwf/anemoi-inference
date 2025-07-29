# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("converter")
class ConverterRunner(DefaultRunner):
    """This runner is used for models that don't predict future states but transform the input data.
    Valid dates at the input are converted into the same valid dates at the output.
    In this configuration, the initial conditions are not written to the output. Instead, step 0 is considered a model output.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.write_initial_state:
            LOG.warning(
                "ConverterRunner does not write initial state to the output. Step 0 is a model output. Setting write_initial_state to False."
            )
            self.write_initial_state = False

    def forecast_stepper(self, start_date, lead_time):
        steps = (lead_time // self.time_step) + 1  # include step 0

        LOG.info("Lead time: %s, time stepping: %s Forecasting %s steps", lead_time, self.time_step, steps)

        for s in range(steps):
            step = s * self.time_step
            valid_date = start_date + step
            next_date = valid_date + self.time_step
            is_last_step = s == steps - 1
            yield step, valid_date, next_date, is_last_step
