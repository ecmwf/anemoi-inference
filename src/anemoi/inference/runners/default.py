# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
from typing import Any

from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from pydantic import BaseModel

from anemoi.inference.config import Configuration
from anemoi.inference.input import Input
from anemoi.inference.modifiers import Modifier
from anemoi.inference.output import Output
from anemoi.inference.processor import Processor
from anemoi.inference.types import IntArray

from ..forcings import BoundaryForcings
from ..forcings import ComputedForcings
from ..forcings import ConstantForcings
from ..forcings import CoupledForcings
from ..forcings import Forcings
from ..inputs import create_input
from ..modifiers import create_modifier
from ..outputs import create_output
from ..post_processors import create_post_processor
from ..pre_processors import create_pre_processor
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


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
        self.reference_date = self.config.date if hasattr(self.config, "date") else None

        super().__init__(
            config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            use_grib_paramid=config.use_grib_paramid,
            patch_metadata=config.patch_metadata,
            development_hacks=config.development_hacks,
            output_frequency=config.output_frequency,
            write_initial_state=config.write_initial_state,
            initial_state_categories=config.initial_state_categories,
            trace_path=config.trace_path,
            use_profiler=config.use_profiler,
            typed_variables=config.typed_variables,
        )

    def predict_step(
        self, model: "torch.nn.Module", input_tensor_torch: "torch.Tensor", **kwargs: Any
    ) -> "torch.Tensor":
        for key, value in self.config.predict_kwargs.items():
            if key in kwargs:
                warnings.warn(
                    f"`predict_kwargs` contains illegal kwarg `{key}`. This kwarg is set by the runner and will be ignored."
                )
                continue
            kwargs[key] = value

        return super().predict_step(model, input_tensor_torch, **kwargs)

    def execute(self) -> None:
        """Execute the runner."""

        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        lead_time = to_timedelta(self.config.lead_time)

        # This may be used by Output objects to compute the step
        self.lead_time = lead_time
        self.time_step = self.checkpoint.timestep

        output = self.create_output()

        # In case the constant forcings are from another input, combine them here
        # So that they are in considered in the `write_initial_state`

        prognostic_input = self.create_prognostics_input()
        LOG.info(f"📥 Prognostic input: {prognostic_input}")
        prognostic_state = prognostic_input.create_input_state(date=self.config.date)
        self._check_state(prognostic_state, "prognostics")

        constants_input = self.create_constant_coupled_forcings_input()
        LOG.info(f"📥 Constant forcings input: {constants_input}")
        constants_state = constants_input.create_input_state(date=self.config.date)
        self._check_state(constants_state, "constant_forcings")

        forcings_input = self.create_dynamic_forcings_input()
        LOG.info(f"📥 Dynamic forcings input: {forcings_input}")
        forcings_state = forcings_input.create_input_state(date=self.config.date)
        self._check_state(forcings_state, "dynamic_forcings")

        input_state = self._combine_states(
            prognostic_state,
            constants_state,
            forcings_state,
        )

        # This hook is needed for the coupled runner
        self.input_state_hook(constants_state)

        # For step-zero only
        initial_state = Output.reduce(
            self._initial_state(
                prognostic_state,
                constants_state,
                forcings_state,
            )
        )
        # Top-level post-processors on the other hand are applied on State and are executed here.
        LOG.info("Top-level post-processors: %s", self.post_processors)

        for processor in self.post_processors:
            initial_state = processor.process(initial_state)

        output.open(initial_state)

        LOG.info("write_initial_state: %s", output)
        output.write_initial_state(initial_state)

        for state in self.run(input_state=input_state, lead_time=lead_time):
            # Apply top-level post-processors
            for processor in self.post_processors:
                state = processor.process(state)
            output.write_state(state)

        output.close()

        if "accumulate_from_start_of_forecast" not in self.config.post_processors:
            LOG.warning(
                """
                🚧 The default accumulation behaviour has changed. 🚧
                🚧 Accumulation fields have NOT been accumulated from the beginning of the forecast. 🚧
                🚧 To accumulate from the beginning, set `post_processors: [accumulate_from_start_of_forecast]` 🚧
                """  # ecmwf/anemoi-inference#131
            )

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

    def create_constant_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:

        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:

        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def _input_forcings(self, *names: str) -> dict[str, Any]:
        """Get the input forcings configuration.

        Parameters
        ----------
        names : list
            The name of the forcings configuration.

        Returns
        -------
        dict
            The input forcings configuration.
        """

        for name in names:
            if name.startswith("-"):
                deprecated = True
                name = name[1:]  # Remove the leading dash
            else:
                deprecated = False
            if self.config.get(name):
                if deprecated:
                    LOG.warning(
                        f"🚫 The `{name}` input forcings configuration is deprecated. "
                        f"Please use the `{names[0]}` configuration instead."
                    )
                if name != names[0]:
                    LOG.info(f"Loading `config.{names[0]}` from `config.{name}`")

                return self.config[name]

        return self.config.input

    #########################################################################################################
    def create_prognostics_input(self) -> Input:
        """Create the prognostics input.

        Returns
        -------
        Input
            The created prognostics input.
        """
        variables = self.variables.retrieved_prognostic_variables()
        config = self._input_forcings("prognostic_input", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="prognostics")
        LOG.info("Prognostic input: %s", input)
        return input

    def create_constant_coupled_forcings_input(self) -> Input:
        """Create the constant coupled forcings input.

        Returns
        -------
        Input
            The created constant coupled forcings input.
        """
        variables = self.variables.retrieved_constant_forcings_variables()
        config = self._input_forcings("constant_forcings", "forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="constant_forcings")
        LOG.info("Constant coupled forcings input: %s", input)
        return input

    def create_dynamic_forcings_input(self) -> Input:
        """Create the dynamic forcings input.

        Returns
        -------
        Input
            The created dynamic forcings input.
        """
        variables = self.variables.retrieved_dynamic_forcings_variables()
        config = self._input_forcings("dynamic_forcings", "-forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="dynamic_forcings")
        LOG.info("Dynamic forcings input: %s", input)
        return input

    def create_boundary_forcings_input(self) -> Input:
        """Create the boundary forcings input.

        Returns
        -------
        Input
            The created boundary forcings input.
        """
        variables = self.variables.retrieved_boundary_forcings_variables()
        config = self._input_forcings("boundary_forcings", "-boundary", "forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="boundary_forcings")
        LOG.info("Boundary forcings input: %s", input)
        return input

    #########################################################################################################
    def create_constant_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
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
        input = self.create_constant_coupled_forcings_input()
        result = ConstantForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)

        return [result]

    def create_dynamic_coupled_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
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
        input = self.create_dynamic_forcings_input()
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return [result]

    def create_boundary_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
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
        input = self.create_boundary_forcings_input()
        result = BoundaryForcings(self, input, variables, mask)
        LOG.info("Boundary forcing: %s", result)
        return [result]

    def create_pre_processors(self) -> list[Processor]:
        """Create pre-processors.

        Returns
        -------
        List[Processor]
            The created pre-processors.
        """
        result = []
        for processor in self.config.pre_processors:
            result.append(create_pre_processor(self, processor))

        LOG.info("Pre processors: %s", result)
        return result

    def create_post_processors(self) -> list[Processor]:
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

    def create_model_modifiers(self) -> list[Modifier]:
        result = []
        for modifier in self.config.model_modifiers:
            result.append(create_modifier(self, modifier))
        LOG.info("Model modifiers: %s", result)
        return result

    def _combine_states(self, *states: dict[str, Any]) -> dict[str, Any]:
        """Combine multiple states into one.

        Parameters
        ----------
        states : list
            The states to combine.

        Returns
        -------
        dict
            The combined state.
        """
        import numpy as np

        combined = states[0].copy()
        combined["fields"] = combined["fields"].copy()
        shape = None
        first_input = combined.get("_input")

        for state in states[1:]:

            this_input = state.get("_input")

            for name, values in itertools.chain(combined["fields"].items(), state.get("fields", {}).items()):
                if shape is None:
                    shape = values.shape
                elif shape != values.shape:
                    raise ValueError(
                        f"Field '{name}' has different shape in the states: "
                        f"{shape} and {values.shape}."
                        f" Input: {first_input} vs {this_input}."
                    )

            if not set(combined["fields"]).isdisjoint(state["fields"]):
                raise ValueError(
                    f"Some states have overlapping fields:"
                    f" {set(combined['fields']).intersection(state['fields'])}"
                    f" Input: {first_input} vs {this_input}."
                )

            combined["fields"].update(state.get("fields", {}))
            for key, value in state.items():
                if key == "fields":
                    continue

                if key.startswith("_"):
                    continue

                if combined.get(key) is None:
                    combined[key] = value
                    continue

                if value is None:
                    continue

                if type(combined[key]) is not type(value):
                    raise ValueError(
                        f"Key '{key}' has different types in the states: " f"{type(combined[key])} and {type(value)}."
                    )

                if isinstance(value, np.ndarray) and isinstance(combined[key], np.ndarray):
                    if not np.array_equal(combined[key], value):
                        raise ValueError(
                            f"Key '{key}' has different array values in the states: "
                            f"{combined[key]} and {value}."
                            f" Input: {first_input} vs {this_input}."
                        )
                    continue

                if combined[key] != value:
                    raise ValueError(
                        f"Key '{key}' has different values in the states: "
                        f"{combined[key]} and {value} ({shape})."
                        f" Input: {first_input} vs {this_input}."
                    )

        return combined

    def _initial_state(
        self, prognostic_state: dict[str, Any], constants_state: dict[str, Any], forcings_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Create the initial state for the output.

        Parameters
        ----------
        prognostic_state : dict
            The prognostic state.
        constants_state : dict
            The constant forcings state.
        forcings_state : dict
            The dynamic forcings state.

        Returns
        -------
        dict
            The initial state for the output.
        """
        states = []

        if "prognostics" in self.config.initial_state_categories:
            states.append(prognostic_state)

        if "constant_forcings" in self.config.initial_state_categories:
            states.append(constants_state)

        if "dynamic_forcings" in self.config.initial_state_categories:
            states.append(forcings_state)

        return self._combine_states(*states)

    def _check_state(self, state: dict[str, Any], title: str) -> None:
        """Check the state for consistency.

        Parameters
        ----------
        state : dict
            The state to check.
        title : str
            The title of the state (for logging).

        Raises
        ------
        ValueError
            If the state is not consistent.
        """

        if not isinstance(state, dict):
            raise ValueError(f"State '{title}' is not a dictionary: {state}")

        input = state.get("_input")

        if "fields" not in state:
            raise ValueError(f"State '{title}' does not contain 'fields': {state} ({input=})")

        shape = None

        for field, values in state["fields"].items():
            if shape is None:
                shape = values.shape
            elif shape != values.shape:
                raise ValueError(
                    f"Field '{field}' in state '{title}' has different shape: "
                    f"{shape} and {values.shape} ({input=})."
                )

        date = state.get("date")
        if date is None and len(state["fields"]) > 0:
            # date can be None for an empty input
            if not isinstance(date, datetime.datetime):
                raise ValueError(f"State '{title}' does not contain 'date', or it is not a datetime: {date} ({input=})")
