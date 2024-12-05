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

from ..forcings import BoundaryForcings
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
            use_grib_paramid=config.use_grib_paramid,
            patch_metadata=config.patch_metadata,
            development_hacks=config.development_hacks,
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
    # Here, by default, we may use the same input "class" as the input
    # not the same instance. This means that we may call mars several times

    def _input_forcings(self, name):
        if self.config.forcings is None:
            # Use the same as the input
            return self.config.input

        if name in self.config.forcings:
            return self.config.forcings[name]

        if "input" in self.config.forcings:
            return self.config.forcings.input

        return self.config.forcings

    def create_constant_coupled_forcings(self, variables, mask):
        input = create_input(self, self._input_forcings("constant"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return result

    def create_dynamic_coupled_forcings(self, variables, mask):

        input = create_input(self, self._input_forcings("dynamic"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return result

    def create_boundary_forcings(self, variables, mask):

        input = create_input(self, self._input_forcings("boundary"))
        result = BoundaryForcings(self, input, variables, mask)
        LOG.info("Boundary forcing: %s", result)
        return result
