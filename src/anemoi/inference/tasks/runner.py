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

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.output import Output
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.runners.testing import TestingMixing
from anemoi.inference.transport import Coupling
from anemoi.inference.transport import Transport
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


class CouplingForcings(CoupledForcings):
    """Just to have a different __repr__."""


class CoupledRunner(DefaultRunner):
    """Runner for coupled models.

    This class handles the initialization and running of coupled models
    using the provided configuration and input.
    """

    def __init__(self, config: dict[str, Any], coupled_input: "CoupledInput") -> None:
        """Initialize the CoupledRunner.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        coupled_input : CoupledInput
            Coupled input instance.
        """
        super().__init__(config)
        self.coupled_input = coupled_input

    def input_state_hook(self, input_state: State) -> None:
        """Hook used by coupled runners to send the input state."""
        self.coupled_input.initial_state(Output.reduce(input_state))

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: Any) -> list[CoupledForcings]:
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
        result = CouplingForcings(self, self.coupled_input, variables, mask)
        return [result]

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: list[Forcings]) -> list[Forcings]:
        """Modify the dynamic forcings inputs for the first step.

        Parameters
        ----------
        dynamic_forcings_inputs : list of Forcings
            The dynamic forcings inputs.

        Returns
        -------
        list[Forcings]
            The modified dynamic forcings inputs.
        """
        # For the initial state we need to load the forcings
        # from the default input.
        result = []
        for f in dynamic_forcings_inputs:
            if isinstance(f, CoupledForcings):
                result.extend(super().create_dynamic_coupled_forcings(f.variables, f.mask))
            else:
                result.append(f)
        return result

    def create_dynamic_forcings_inputs(self, input_state: State) -> list[Forcings]:
        """Create dynamic forcings inputs.

        Parameters
        ----------
        input_state : State
            The input state.

        Returns
        -------
        list[Forcings]
            The created dynamic forcings inputs.
        """
        forcings = super().create_dynamic_forcings_inputs(input_state)
        result = []
        for f in forcings:
            if isinstance(f, CoupledForcings):
                # Substituting the CoupledForcings input with the coupled input
                # TODO: review this
                f = CouplingForcings(self, self.coupled_input, f.variables, f.mask)

            result.append(f)

        return result


class CoupledInput:
    """Input handler for coupled models.

    This class manages the input data and state for coupled models,
    including loading and initializing forcings.
    """

    trace_name = "coupled"

    def __init__(self, task: Task, transport: Any, couplings: list[Coupling]) -> None:
        """Initialize the CoupledInput.

        Parameters
        ----------
        task : Task
            Task instance.
        transport : Any
            Transport instance.
        couplings : list of Coupling
            List of couplings.
        """
        self.task = task
        self.transport = transport
        self.couplings = couplings
        self.constants: dict[str, FloatArray] = {}
        self.tag = 0

    def load_forcings_state(self, *, variables: list[str], dates: list[Date], current_state: State) -> State:
        """Load the forcings state.

        Parameters
        ----------
        variables : list of str
            List of variable names.
        dates : list of Date
            List of dates.
        current_state : State
            Current state dictionary.

        Returns
        -------
        State
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


class TestCoupledRunner(TestingMixing, CoupledRunner):
    """Runner for testing coupled models."""

    def __init__(self, config: dict[str, Any], coupled_input: "CoupledInput") -> None:
        """Initialize the TestCoupledRunner.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        coupled_input : CoupledInput
            Coupled input instance.
        """

        super().__init__(config, coupled_input)


@task_registry.register("runner")
class RunnerTask(Task):
    """Task for running coupled models.

    This task initializes and runs coupled models using the provided
    configuration, overrides, and global configuration.
    """

    def __init__(
        self, name: str, config: dict[str, Any], overrides: dict[str, Any] = {}, global_config: dict[str, Any] = {}
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

        assert self.config.runner in ("default", "testing"), self.config.runner

        coupler = CoupledInput(self, transport, couplings)

        if self.config.runner == "testing":
            runner = TestCoupledRunner(self.config, coupler)
        else:
            runner = CoupledRunner(self.config, coupler)

        runner.execute()
