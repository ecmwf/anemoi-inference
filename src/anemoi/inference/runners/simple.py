# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import Forcings
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.types import IntArray

from . import runner_registry

LOG = logging.getLogger(__name__)


class SimpleTensorHandler(TensorHandler):
    """This tensor handler only supports computed forcings."""

    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        LOG.warning(f"[{self.name}] Constant forcings are not supported by this runner.")
        if variables:
            LOG.warning(f"[{self.name}] {variables} must be provided in the input state by the user.")
        return []

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        LOG.warning(f"[{self.name}] Dynamic forcings are not supported by this runner.")
        if variables:
            LOG.warning(
                f"[{self.name}] {variables} must be provided in the input state and updated during rollout by the user."
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
            checkpoint=checkpoint, input=kwargs.pop("input", "empty"), output=kwargs.pop("output", "none"), **kwargs
        )
        super().__init__(config, classes=RunnerClasses(tensor_handler=SimpleTensorHandler))

    def execute(self) -> None:
        raise NotImplementedError(
            "The `execute` method is not supported by SimpleRunner. Use the `run` generator instead."
        )
