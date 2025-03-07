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

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.output import Output
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.transport import Transport
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


class CoupledRunner(DefaultRunner):
    """Runner for coupled models.

    This class handles the initialization and running of coupled models
    using the provided configuration and input.
    """

    def __init__(self, config: Dict[str, Any], coupled_input: "CoupledInput") -> None:
        """Initialize the CoupledRunner.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        input : CoupledInput
            Coupled input instance.
        """
        super().__init__(config)
        self.coupled_input = coupled_input

    def input_state_hook(self, input_state: State) -> None:
        """Hook used by coupled runners to send the input state."""
        self.coupled_input.initial_state(Output.reduce(input_state))

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
        result = CoupledForcings(self, self.coupled_input, variables, mask)
        return [result]

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: List[Forcings]) -> List[Forcings]:
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

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
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

    def initial_state(self, state: State) -> None:
        """Initialize the state.

        Parameters
        ----------
        state : State
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
        self.config = RunConfiguration.load(config, overrides=[global_config, overrides])

    def run(self, transport: Transport) -> None:
        """Run the task.

        Parameters
        ----------
        transport : Transport
            Transport instance.
        """
        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self)

        coupler = CoupledInput(self, transport, couplings)
        runner = CoupledRunner(self.config, coupler)
        runner.execute()
