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
    """Runner for coupled models.

    This class handles the initialization and running of coupled models
    using the provided configuration and input.
    """

    def __init__(self, config: Dict[str, Any], input: "CoupledInput") -> None:
        """Initialize the CoupledRunner.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        input : CoupledInput
            Coupled input instance.
        """
        super().__init__(config)
        self.input = input

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: Any) -> List[CoupledForcings]:
        """Create dynamic coupled forcings.

        Parameters
        ----------
        variables : list of str
            List of variable names.
        mask : Any
            Mask to apply to the variables.

        Returns
        -------
        list of CoupledForcings
            List of coupled forcings.
        """
        result = CoupledForcings(self, self.input, variables, mask)
        return [result]

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: List[Dict[str, Any]]) -> List[CoupledForcings]:
        """Initialize dynamic forcings inputs.

        Parameters
        ----------
        dynamic_forcings_inputs : list of dict
            List of dynamic forcings input dictionaries.

        Returns
        -------
        list of CoupledForcings
            List of coupled forcings.
        """
        result = []
        for c in dynamic_forcings_inputs:
            result.extend(super().create_dynamic_coupled_forcings(c.variables, c.mask))
        return result


class CoupledInput:
    """Input handler for coupled models.

    This class manages the input data and state for coupled models,
    including loading and initializing forcings.
    """

    trace_name = "coupled"

    def __init__(self, task: Task, transport: Any, couplings: List[Any]) -> None:
        """Initialize the CoupledInput.

        Parameters
        ----------
        task : Task
            Task instance.
        transport : Any
            Transport instance.
        couplings : list of Any
            List of couplings.
        """
        self.task = task
        self.transport = transport
        self.couplings = couplings
        self.constants = {}
        self.tag = 0

    def load_forcings_state(
        self, *, variables: List[str], dates: List[str], current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load the forcings state.

        Parameters
        ----------
        variables : list of str
            List of variable names.
        dates : list of str
            List of dates.
        current_state : dict
            Current state dictionary.

        Returns
        -------
        dict
            Updated state dictionary.
        """
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
        """Initialize the state.

        Parameters
        ----------
        state : dict
            State dictionary.
        """
        # We want to copy the constants that may be requested by the other tasks
        # For now, we keep it simple and just copy the whole state
        self.constants = state["fields"].copy()


@task_registry.register("runner")
class RunnerTask(Task):
    """Task for running coupled models.

    This task initializes and runs coupled models using the provided
    configuration, overrides, and global configuration.
    """

    def __init__(
        self, name: str, config: Dict[str, Any], overrides: Dict[str, Any] = {}, global_config: Dict[str, Any] = {}
    ) -> None:
        """Initialize the RunnerTask.

        Parameters
        ----------
        name : str
            Name of the task.
        config : dict
            Configuration dictionary.
        overrides : dict, optional
            Overrides dictionary.
        global_config : dict, optional
            Global configuration dictionary.
        """
        super().__init__(name)
        LOG.info("Creating RunnerTask %s %s (%s)", self, config, global_config)
        self.config = CoupleConfiguration.load(config, overrides=[global_config, overrides])

    def run(self, transport: Any) -> None:
        """Run the task.

        Parameters
        ----------
        transport : Any
            Transport instance.
        """
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
