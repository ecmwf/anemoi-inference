# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.inference.config import load_config
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.runners.default import DefaultRunner

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


class CoupledRunner(DefaultRunner):
    """_summary_"""

    def __init__(self, config, input):
        super().__init__(config)
        self.input = input

    def create_dynamic_coupled_forcings(self, variables, mask):
        result = CoupledForcings(self, self.input, variables, mask)
        return [result]


class CoupledInput:
    """_summary_"""

    def __init__(self, task, transport, couplings):
        self.task = task
        self.transport = transport
        self.couplings = couplings

    def load_forcings(self, variables, dates):
        # self.transport.receive(self.task, data)
        pass


@task_registry.register("runner")
class RunnerTask(Task):
    """_summary_"""

    def __init__(self, name, config):
        super().__init__(name)
        LOG.info("Creating RunnerTask %s %s", self, config)
        self.config = load_config(config, [])

    def run(self, transport):
        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self)

        runner = CoupledRunner(self.config, CoupledInput(self, transport, couplings))
        input = runner.create_input()
        output = runner.create_output()

        input_state = input.create_input_state(date=self.config.date)

        if self.config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=self.config.lead_time):
            output.write_state(state)

        output.close()
