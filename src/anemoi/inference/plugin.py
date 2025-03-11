# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This is an `ai-models` plugin that can run anemoi.inference models."""

import argparse
import logging
import os
from functools import cached_property
from typing import Any
from typing import List
from typing import Optional

from ai_models.model import Model

from anemoi.inference.inputs.grib import GribInput
from anemoi.inference.outputs.grib import GribOutput
from anemoi.inference.runner import PRECISIONS as AUTOCAST
from anemoi.inference.runners.plugin import PluginRunner
from anemoi.inference.types import Date
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


class FieldListInput(GribInput):
    """Handles earchkit-data fieldlists input fields."""

    def __init__(self, context: Any, *, input_fields: Any) -> None:
        """Initialize FieldListInput.

        Parameters
        ----------
        context : Any
            The context for the input.
        input_fields : Any
            The input fields to be processed.
        """
        super().__init__(context)
        self.input_fields = input_fields

    def create_input_state(self, *, date: Optional[Date]) -> Any:
        """Create the input state for the given date.

        Parameters
        ----------
        date : str
            The date for which to create the input state.

        Returns
        -------
        Any
            The created input state.
        """
        return self._create_input_state(self.input_fields, variables=None, date=date)

    def load_forcings_state(
        self,
        *,
        variables: List[str],
        dates: List[str],
        current_state: State,
    ) -> State:
        """Load the forcings state.

        Parameters
        ----------
        variables : List[str]
            The variables to load.
        dates : List[str]
            The dates for which to load the forcings.
        current_state : State
            The current state to update.

        Returns
        -------
        State
            The updated state with loaded forcings.
        """
        return self._load_forcings_state(
            self.input_fields,
            variables=variables,
            dates=dates,
            current_state=current_state,
        )

    def set_private_attributes(self, state: State, input_fields: Any) -> None:
        """Set private attributes for the state.

        Parameters
        ----------
        state : Dict[str, Any]
            The state to update.
        input_fields : Any
            The input fields to use for setting attributes.
        """
        input_fields = input_fields.order_by("valid_datetime")
        state["_grib_templates_for_output"] = {field.metadata("name"): field for field in input_fields}


class CallbackOutput(GribOutput):
    """Call ai-models write method."""

    def __init__(self, context: Any, *, write: Any, encoding: Any = None) -> None:
        """Initialize CallbackOutput.

        Parameters
        ----------
        context : Any
            The context for the output.
        write : Any
            The write method to call.
        encoding : Any, optional
            The encoding to use, by default None.
        """
        super().__init__(context, encoding=encoding, templates={"source": "templates"})
        self.write = write

    def write_message(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Write a message using the write method.

        Parameters
        ----------
        message : Any
            The message to write.
        args : Any
            Additional arguments.
        kwargs : Any
            Additional keyword arguments.
        """
        self.write(message, *args, **kwargs)


class AIModelPlugin(Model):

    expver: Any = None

    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        """To be implemented in subclasses to add model-specific arguments to the parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            An instance of the parser to add arguments to.
        """
        pass

    def parse_model_args(self, args: List[str]) -> argparse.ArgumentParser:
        """Parse model-specific arguments.

        Parameters
        ----------
        args : List[str]
            The list of arguments to parse.

        Returns
        -------
        argparse.ArgumentParser
            The parser with parsed arguments.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--checkpoint", required=not hasattr(self, "download_files"))
        parser.add_argument(
            "--autocast",
            type=str,
            choices=sorted(AUTOCAST.keys()),
        )

        self.add_model_args(parser)

        args = parser.parse_args(args)
        args._checkpoint = args.checkpoint

        if args._checkpoint is None:
            args._checkpoint = os.path.join(self.assets, self.download_files[0])

        for k, v in vars(args).items():
            setattr(self, k, v)

        return parser

    @cached_property
    def runner(self) -> PluginRunner:
        """Get the PluginRunner instance."""
        return PluginRunner(self._checkpoint, device=self.device)

    def run(self) -> None:
        """Run the model inference."""
        if self.deterministic:
            self.torch_deterministic_mode()

        input_kwargs = self.input.anemoi_plugin_input_kwargs()
        output_kwargs = self.input.anemoi_plugin_input_kwargs()

        input = FieldListInput(self.runner, input_fields=self.all_fields, **input_kwargs)
        output = CallbackOutput(self.runner, write=self.write, **output_kwargs)

        input_state = input.create_input_state(date=self.start_datetime)

        output.write_initial_state(input_state)

        for state in self.runner.run(input_state=input_state, lead_time=self.lead_time):
            output.write_state(state)

        output.close()

    # Below are methods forwarded to the checkpoint

    @property
    def param_sfc(self) -> Any:
        """Surface parameters."""
        return self.runner.param_sfc

    @property
    def param_level_pl(self) -> Any:
        """Pressure level parameters."""
        return self.runner.param_level_pl

    @property
    def param_level_ml(self) -> Any:
        """Model level parameters."""
        return self.runner.param_level_ml

    @property
    def constant_fields(self) -> Any:
        """Constant fields from input."""
        return self.runner.checkpoint.constants_from_input

    @property
    def grid(self) -> Any:
        """Grid information."""
        return self.runner.checkpoint.grid

    @property
    def area(self) -> Any:
        """Area information."""
        return self.runner.checkpoint.area

    @property
    def lagged(self) -> Any:
        """Lagged information."""
        return self.runner.lagged


model = AIModelPlugin
