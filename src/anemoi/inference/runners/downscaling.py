# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from copy import deepcopy

from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from pydantic import BaseModel

from ..checkpoint import Checkpoint
from ..runner import Runner
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class Low(DefaultRunner):
    pass


class High(DefaultRunner):
    pass


@runner_registry.register("downscaling")
class DownscalingRunner(Runner):
    """Runner from a configuration file."""

    def __init__(self, config):

        if isinstance(config, dict):
            # So we get the dot notation
            config = DotDict(config)

        # Remove that when the Pydantic model is ready
        if isinstance(config, BaseModel):
            config = DotDict(config.model_dump())

        self.config = config
        self.forcings = None  # A 'Checkpoint' object that points to the forcings

        low_checkpoint, hig_checkpoint = Checkpoint(config.checkpoint).split()

        lowc = deepcopy(config)
        lowc.checkpoint = low_checkpoint

        higc = deepcopy(config)
        higc.checkpoint = hig_checkpoint

        self.low_runner = Low(lowc)
        self.high_runner = High(higc)

        super().__init__(
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
            use_grib_paramid=config.use_grib_paramid,
            patch_metadata=config.patch_metadata,
            development_hacks=config.development_hacks,
        )

    def inference_loop(self, start, lead_time):
        LOG.info("Inference loop: start=%s, lead_time=%s", start, lead_time)
        # return self.checkpoint.inference_loop(start, leadjson_time)
        for s in range(10):
            step = (s + 1) * self.checkpoint.timestep
            date = start + step
            LOG.info("Forecasting step %s (%s)", step, date)
            yield date, step, False

    def execute(self, date, lead_time, write_initial_state):

        high_input = self.high_runner.create_input()
        high_input_state = high_input.create_input_state(date=date)
        print("✅✅✅✅✅✅✅✅✅--------x", list(high_input_state["fields"].keys()))
        self.high_runner.add_forcings(high_input_state)
        print("✅✅✅✅✅✅✅✅✅--------y", list(high_input_state["fields"].keys()))
        high_input_tensor = self.high_runner.prepare_input_tensor(high_input_state)
        print("✅✅✅✅✅✅✅✅✅--------z", list(high_input_state["fields"].keys()))

        assert False, "Not implemented"

        low_input = self.low_runner.create_input()
        low_input_state = low_input.create_input_state(date=date)
        self.low_runner.add_forcings(low_input_state)
        low_input_tensor = self.low_runner.prepare_input_tensor(low_input_state)

        assert False, "Not implemented"
        # for state in self.run(input_state=(low_input_state,high_input_state) , lead_time=lead_time):
        #     # output.write_state(state)
        #     pass

        # input = (self.low_runner.create_input(), self.high_runner.create_input())
        # output = self.low_runner.create_output()

        # input_state_low = input[0].create_input_state(date=date)
        # input_state_highres = input[1].create_input_state(date=date)

        # for state in self.run(input_state=(input_state_low, input_state_highres), lead_time=lead_time):
        #     output.write_state(state)

        # # output.close()

    def run(self, *, input_state, lead_time):

        # timers = Timers()

        lead_time = to_timedelta(lead_time)

        # This may be used but Output objects to compute the step
        self.lead_time = lead_time
        self.time_step = self.checkpoint.timestep

        input_tensors = [self.prepare_input_tensor(x) for x in input_state]

        try:
            yield from self.postprocess(self.forecast(lead_time, input_tensors, input_state))
        except (TypeError, ModuleNotFoundError, AttributeError):
            if self.report_error:
                self.checkpoint.report_error()
            raise
