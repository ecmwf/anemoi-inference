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
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.runners.testing import NoModelMixing
from anemoi.inference.state import reduce_state
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.transport import Coupling
from anemoi.inference.transport import Transport
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


class CouplingForcings(CoupledForcings):
    """Just to have a different __repr__."""

    def __init__(self, context, input, variables, mask):
        super().__init__(context, input, variables, mask)
        self.kinds = dict(coupled=True)


class CoupledTensorHandler(TensorHandler):
    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[CoupledForcings]:
        mine = []
        other = []

        for i, v in enumerate(variables):
            if v in self.context.coupled_input.variables_to_recieve:
                mine.append(i)
            else:
                other.append(i)

        result = []
        if mine:
            result.append(CouplingForcings(self, self.context.coupled_input, [variables[i] for i in mine], mask[mine]))

        if other:
            result.extend(super().create_dynamic_coupled_forcings([variables[i] for i in other], mask[other]))

        return result

    def initial_dynamic_forcings_providers(self, dynamic_forcings_providers: list[Forcings]) -> list[Forcings]:
        """Modify the dynamic forcings providers for the first step.
        For the initial state we need to load the dynamic coupled forcings from the default provider (i.e.: from disk).
        """

        result = []
        for f in dynamic_forcings_providers:
            if isinstance(f, CoupledForcings):
                result.extend(super().create_dynamic_coupled_forcings(f.variables, f.mask))
            else:
                result.append(f)
        return result


class CoupledRunner(Runner):
    """Runner for coupled models.

    This class handles the initialization and running of coupled models
    using the provided configuration and input.
    """

    def __init__(self, config: RunConfiguration, coupled_input: "CoupledInput") -> None:
        """Initialize the CoupledRunner."""
        self.coupled_input = coupled_input
        super().__init__(
            config,
            classes=RunnerClasses(tensor_handler=CoupledTensorHandler),
        )

        if len(self.checkpoint.multi_dataset_metadata) > 1:
            LOG.warning(
                "Coupling models with multiple datasets is not yet fully supported and may lead to unexpected behaviour."
                "Coupling variables cross-dataset is not supported."
            )

    def input_states_hook(self, input_states: dict[str, State]) -> None:
        """Hook used by coupled runners to send the input state."""
        for state in input_states.values():
            self.coupled_input.initial_state(reduce_state(state))

    def output_states_hook(self, output_states: dict[str, State]) -> None:
        """Hook used by coupled runners to send the input state."""
        if self.checkpoint.multi_step_output > 1:
            raise NotImplementedError(
                "output_state_hook is not yet supported when multi-step output and coupled runners are combined"
            )
        for state in output_states.values():
            self.coupled_input.output_state(reduce_state(state))


class CoupledInput:
    """Input handler for coupled models.

    This class manages the input data and state for coupled models,
    including loading and initializing forcings.
    """

    trace_name = "coupled"

    def __init__(self, task: Task, transport: Transport, couplings: list[Coupling]):
        self.task = task
        self.transport = transport
        self.couplings = couplings
        self.constants: dict[str, FloatArray] = {}
        self.tag = 0
        self.send_only = sum(1 if c.target is task else 0 for c in couplings) == 0
        self.variables_to_recieve = set(sum([c.variables for c in couplings if c.target is task], []))

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        LOG.info(f"Adding dynamic forcings {dates}")
        state = dict(date=dates)

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
        # We want to copy the constants that may be requested by the other tasks
        # For now, we keep it simple and just copy the whole state

        fields: dict = state["fields"]

        common_fields = set(fields.keys()) & set(self.constants.keys())
        if common_fields:
            # with multi-dataset we currently don't support cross-dataset
            raise RuntimeError(
                f"Fields {common_fields} are already coupled. Multi-dataset coupling is not yet supported."
            )

        self.constants.update(fields.copy())

    def output_state(self, state: State) -> None:
        if not self.send_only:
            return

        # We exchange the states when fetching forcings from the coupled tasks.
        # If this task is send only, it will never exchange its state, so
        # we do it here.

        for c in self.couplings:
            c.apply(
                self.task,
                self.transport,
                input_state=state,
                output_state=None,
                constants=self.constants,
                tag=self.tag,
            )

        self.tag += 1


class NoModelCoupledRunner(NoModelMixing, CoupledRunner):
    """Runner for testing coupled models."""

    def __init__(self, config: RunConfiguration, coupled_input: "CoupledInput") -> None:
        super().__init__(config, coupled_input)


@task_registry.register("runner")
class RunnerTask(Task):
    """Task for running coupled models.

    This task initializes and runs coupled models using the provided
    configuration, overrides, and global configuration.
    """

    def __init__(
        self, task_name: str, config: dict[str, Any], overrides: dict[str, Any] = {}, global_config: dict[str, Any] = {}
    ) -> None:
        super().__init__(task_name)
        LOG.info("Creating RunnerTask %s %s (%s)", self, config, global_config)
        self.config = RunConfiguration.load(config, overrides=[global_config, overrides])

    def run(self, transport: Transport) -> None:
        """Run the task."""

        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self)

        assert self.config.runner in ("default", "no-model"), self.config.runner

        coupler = CoupledInput(self, transport, couplings)

        # TODO: a factory method would be better here
        if self.config.runner == "no-model":
            runner = NoModelCoupledRunner(self.config, coupler)
        else:
            runner = CoupledRunner(self.config, coupler)

        runner.execute()
        LOG.info("Finished task %s", self.name)
