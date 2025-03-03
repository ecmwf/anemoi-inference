# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict
from typing import List

from anemoi.inference.config.couple import CoupleConfiguration
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.runners.default import DefaultRunner

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


class CoupledRunner(DefaultRunner):

    def __init__(self, config: Dict[str, Any], input: "CoupledInput") -> None:
        super().__init__(config)
        self.input = input

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: Any) -> List[CoupledForcings]:
        result = CoupledForcings(self, self.input, variables, mask)
        return [result]

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: List[Dict[str, Any]]) -> List[CoupledForcings]:
        result = []
        for c in dynamic_forcings_inputs:
            result.extend(super().create_dynamic_coupled_forcings(c.variables, c.mask))
        return result


class CoupledInput:

    trace_name = "coupled"

    def __init__(self, task: Task, transport: Any, couplings: List[Any]) -> None:
        self.task = task
        self.transport = transport
        self.couplings = couplings
        self.constants = {}
        self.tag = 0

    def load_forcings_state(
        self, *, variables: List[str], dates: List[str], current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        LOG.info("Adding dynamic forcings %s %s", variables, dates)
        state = dict(variables=variables, date=dates)

        for c in self.couplings:
            c.apply(
                self.task,
                self.transport,
                input_state=current_state,
                output_state=state,
                constants=self.constants,
                tag=self.tag,
            )

        for f, v in state["fields"].items():
            assert len(v.shape) == 1, (f, v.shape)

        assert state["date"] == dates, (state["date"], dates)

        self.tag += 1

        return state

    def initial_state(self, state: Dict[str, Any]) -> None:
        # We want to copy the constants that may be requested by the other tasks
        # For now, we keep it simple and just copy the whole state
        self.constants = state["fields"].copy()


@task_registry.register("runner")
class RunnerTask(Task):

    def __init__(
        self, name: str, config: Dict[str, Any], overrides: Dict[str, Any] = {}, global_config: Dict[str, Any] = {}
    ) -> None:
        super().__init__(name)
        LOG.info("Creating RunnerTask %s %s (%s)", self, config, global_config)
        self.config = CoupleConfiguration.load(config, overrides=[global_config, overrides])

    def run(self, transport: Any) -> None:
        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self)

        coupler = CoupledInput(self, transport, couplings)
        runner = CoupledRunner(self.config, coupler)

        # TODO: Forctorise with the similar code in commands/run.py
        input = runner.create_input()
        output = runner.create_output()

        input_state = input.create_input_state(date=self.config.date)
        coupler.initial_state(output.reduce(input_state))

        if self.config.write_initial_state:
            output.write_initial_state(input_state)

        for state in runner.run(input_state=input_state, lead_time=self.config.lead_time):
            output.write_state(state)

        output.close()
