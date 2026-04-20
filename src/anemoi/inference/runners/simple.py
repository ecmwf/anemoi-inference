# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Generator

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import Forcings
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from . import runner_registry

LOG = logging.getLogger(__name__)


class SimpleTensorHandler(TensorHandler):
    """This tensor handler only supports computed forcings."""

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        LOG.warning(f"[{self.dataset_name}] Constant forcings are not supported by this runner.")
        if variables:
            LOG.warning(f"[{self.dataset_name}] {variables} must be provided in the input state by the user.")
        return []

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        LOG.warning(f"[{self.dataset_name}] Dynamic forcings are not supported by this runner.")
        if variables:
            LOG.warning(
                f"[{self.dataset_name}] {variables} must be provided in the input state and updated during rollout by the user."
            )
        return []


@runner_registry.register("simple")
class SimpleRunner(Runner):
    """Runner for the low-level API.
    The user provides a prepared State object and directly calls the `Runner.run` generator.
    The input State must contain all fields expected by the model, except for computed forcings which will be loaded by the runner.
    The user is responsible for providing any constant and dynamic forcings in the input State and during rollout.
    """

    def __init__(self, checkpoint: str, **kwargs):
        """Initialize the SimpleRunner.

        Parameters
        ----------
        checkpoint : str
            Path to the model checkpoint.
        **kwargs : Any
            Additional keyword arguments passed to the `RunConfiguration`.
        """
        config = RunConfiguration(
            checkpoint=checkpoint,
            input=kwargs.pop("input", "empty"),
            output=kwargs.pop("output", "none"),
            **kwargs,
        )
        super().__init__(config, classes=RunnerClasses(tensor_handler=SimpleTensorHandler))

    def execute(self) -> None:
        raise NotImplementedError(
            "The `execute` method is not supported by SimpleRunner. Use the `run` generator instead."
        )

    def run(
        self, *, input_states: dict[str, State] | State, **kwargs
    ) -> Generator[dict[str, State] | State, None, None]:
        # for multi-dataset checkpoints with more than one dataset, users must provide a dict of States
        # for backwards compatibility also allow users to pass a single State for single-dataset checkpoints

        multi_metadata = self.checkpoint.multi_dataset_metadata
        legacy = False

        if len(multi_metadata) == 1:
            dataset_name = next(iter(multi_metadata))
            if dataset_name not in input_states:
                legacy = True
                input_states = {dataset_name: input_states}
        else:
            for dataset_name in multi_metadata:
                if dataset_name not in input_states:
                    raise ValueError(
                        f"Dataset `{dataset_name}` is expected by the model but not found in the input_states dictionary."
                    )

        for states in super().run(input_states=input_states, **kwargs):
            if legacy:
                # user provided a single State, so we also return a single State
                yield next(iter(states.values()))
            else:
                yield states
