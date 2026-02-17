# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from typing import Any
from typing import Dict
from typing import List

from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from pydantic import BaseModel

from anemoi.inference.config import Configuration
from anemoi.inference.input import Input
from anemoi.inference.output import Output
from anemoi.inference.processor import Processor
from anemoi.inference.types import IntArray
from anemoi.inference.types import State

from ..forcings import BoundaryForcings
from ..forcings import ComputedForcings
from ..forcings import CoupledForcings
from ..forcings import Forcings
from ..inputs import create_input
from ..outputs import create_output
from ..post_processors import create_post_processor
from ..pre_processors import create_pre_processor
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("default")
class DefaultRunner(Runner):
    """Default runner class for inference.

    This class provides the default implementation for running inference.
    """

    def __init__(self, config: Configuration) -> None:
        """Initialize the DefaultRunner.

        Parameters
        ----------
        config : Configuration
            Configuration for the runner.
        """

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
            output_frequency=config.output_frequency,
            write_initial_state=config.write_initial_state,
            trace_path=config.trace_path,
            use_profiler=config.use_profiler,
        )

    def execute(self) -> None:
        """Execute the runner."""

        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        lead_time = to_timedelta(self.config.lead_time)

        # This may be used by Output objects to compute the step
        self.lead_time = lead_time
        self.time_step = self.checkpoint.timestep

        input = self.create_input()
        output = self.create_output()

        # pre_processors = self.pre_processors
        post_processors = self.post_processors

        input_state = input.create_input_state(date=self.config.date)

        # This hook is needed for the coupled runner
        self.input_state_hook(input_state)

        state = Output.reduce(input_state)
        for processor in post_processors:
            state = processor.process(state)

        output.open(state)
        output.write_initial_state(state)

        for state in self.run(input_state=input_state, lead_time=lead_time):
            for processor in post_processors:
                LOG.info("Post processor: %s", processor)
                state = processor.process(state)
            output.write_state(state)

        output.close()

    def input_state_hook(self, input_state: State) -> None:
        """Hook used by coupled runners to send the input state."""
        pass

    def create_input(self) -> Input:
        """Create the input.

        Returns
        -------
        Input
            The created input.
        """
        input = create_input(self, self.config.input)
        LOG.info("Input: %s", input)
        return input

    def create_output(self) -> Output:
        """Create the output.

        Returns
        -------
        Output
            The created output.
        """
        output = create_output(self, self.config.output)
        LOG.info("Output: %s", output)
        return output

    def create_constant_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created dynamic computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def _input_forcings(self, name: str) -> Dict[str, Any]:
        """Get the input forcings configuration.

        Parameters
        ----------
        name : str
            The name of the forcings configuration.

        Returns
        -------
        dict
            The input forcings configuration.
        """
        if self.config.forcings is None:
            # Use the same as the input
            return self.config.input

        if name in self.config.forcings:
            return self.config.forcings[name]

        if "input" in self.config.forcings:
            return self.config.forcings.input

        return self.config.forcings

    def create_constant_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant coupled forcings.

        Parameters
        ----------
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant coupled forcings.
        """
        input = create_input(self, self._input_forcings("constant"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return [result]

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic coupled forcings.

        Parameters
        ----------
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created dynamic coupled forcings.
        """
        input = create_input(self, self._input_forcings("dynamic"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return [result]

    def create_boundary_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create boundary forcings.

        Parameters
        ----------
        variables : list
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created boundary forcings.
        """
        input = create_input(self, self._input_forcings("boundary"))
        result = BoundaryForcings(self, input, variables, mask)
        LOG.info("Boundary forcing: %s", result)
        return [result]

    def create_pre_processors(self) -> List[Processor]:
        """Create pre-processors.

        Returns
        -------
        List[Processor]
            The created pre-processors.
        """
        if self.config.post_processors is None:
            self.config.post_processors = ["accumulate_from_start_of_forecast"]
            warnings.warn(
                """
                No post_processors defined. Accumulations will be accumulated from the beginning of the forecast.

                🚧🚧🚧 In a future release, the default will be to NOT accumulate from the beginning of the forecast. 🚧🚧🚧
                Update your config if you wish to keep accumulating from the beginning.
                https://github.com/ecmwf/anemoi-inference/issues/131
                """,
            )

        if "accumulate_from_start_of_forecast" not in self.config.post_processors:
            warnings.warn(
                """
                post_processors are defined but `accumulate_from_start_of_forecast` is not set."
                🚧 Accumulations will NOT be accumulated from the beginning of the forecast. 🚧
                """
            )

        result = []
        for processor in self.config.pre_processors:
            result.append(create_pre_processor(self, processor))

        LOG.info("Pre processors: %s", result)
        return result

    def create_post_processors(self) -> List[Processor]:
        """Create post-processors.

        Returns
        -------
        List[Processor]
            The created post-processors.
        """
        result = []
        for processor in self.config.post_processors:
            result.append(create_post_processor(self, processor))

        LOG.info("Post processors: %s", result)
        return result
