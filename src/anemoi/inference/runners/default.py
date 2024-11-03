# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.utils.config import DotDict
from pydantic import BaseModel

from ..forcings import ComputedForcings
from ..forcings import CoupledForcings
from ..inputs import create_input
from ..outputs import create_output
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("default")
class DefaultRunner(Runner):
    """Runner from a configuration file."""

    def __init__(self, config):

        if isinstance(config, dict):
            # So we get the dot notation
            config = DotDict(config)

        # Remove that when the Pydantic model is ready
        if isinstance(config, BaseModel):
            config = DotDict(config.model_dump())

        self.config = config

        super().__init__(
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
        )

    def create_input(self):
        input = create_input(self, self.config.input)
        LOG.info("Input: %s", input)
        return input

    def create_output(self):
        output = create_output(self, self.config.output)
        LOG.info("Output: %s", output)
        return output

    # Computed forcings
    def create_constant_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return result

    def create_dynamic_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return result

    # Coupled forcings
    # TODO: Connect them to the Input if needed

    def create_constant_coupled_forcings(self, variables, mask):

        input = self.config.forcings.input
        if "constant" in input:
            input = input.constant

        input = create_input(self, input)
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return result

    def create_dynamic_coupled_forcings(self, variables, mask):

        input = self.config.forcings.input
        if "dynamic" in input:
            input = input.dynamic

        input = create_input(self, input)
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return result
