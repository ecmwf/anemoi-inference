# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from datetime import timedelta
from typing import Any

from anemoi.inference.metadata import Metadata
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.state import reduce_state
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import ForwardOutput
from . import output_registry

LOG = logging.getLogger(__name__)


def create_diagnostic_input(context, metadata, dataset_name):
    from anemoi.inference.config.utils import multi_datasets_config
    from anemoi.inference.inputs import create_input

    variables = metadata.select_variables(
        include=["diagnostic"],
        exclude=["computed", "prognostic", "forcing"],
    )
    config = context.config["input"] if variables else "empty"
    config = multi_datasets_config(config, dataset_name, context.dataset_names)
    return create_input(context, config, metadata, variables=variables, purpose="diagnostics")  # type: ignore[reportArgumentType]


@output_registry.register("truth")
@main_argument("output")
class TruthOutput(ForwardOutput):
    """Write the input state at the same time for each output state.

    Can only be used for inputs with that have access to the time of
    the forecasts, effectively only for times in the past.
    """

    def __init__(
        self, context: DefaultRunner, metadata: Metadata, *, output, add_diagnostic: bool = False, **kwargs: Any
    ) -> None:
        """Initialise the TruthOutput.

        Parameters
        ----------
        context : Context
            The context for the output.
        metadata : Metadata
            Metadata corresponding to the dataset this output is handling.
        output :
            The output configuration.
        add_diagnostic : bool, optional
            Whether to add a diagnostic fields into the truth, by default False.
            Should only be used if the diagnostic fields are available in the input, i.e. datasets.
        kwargs : dict
            Additional keyword arguments.
        """
        if not isinstance(context, DefaultRunner):
            raise ValueError("TruthOutput can only be used with `DefaultRunner`")

        super().__init__(context, metadata, output, None, **kwargs)

        self._prog_input = context.prognostics_inputs[metadata.dataset_name]
        self._diag_input = create_diagnostic_input(context, metadata, metadata.dataset_name) if add_diagnostic else None

        self._dynamic_forc_input = context.dynamic_forcings_inputs[metadata.dataset_name]
        self._constant_forc_input = context.constant_forcings_inputs[metadata.dataset_name]

    def modify_state(self, state: State) -> State:
        """Modify state by overriding it with the truth state."""

        states = [
            self._prog_input.create_input_state(date=state["date"]),
            self._constant_forc_input.create_input_state(date=state["date"]),
            self._dynamic_forc_input.create_input_state(date=state["date"]),
        ]
        if self._diag_input:
            states.append(self._diag_input.create_input_state(date=state["date"]))

        truth_state = self.context._combine_states(*states)  # type: ignore
        truth_state = reduce_state(truth_state)
        truth_state["step"] = state.get("step", timedelta(hours=0))

        return truth_state

    def __repr__(self) -> str:
        """Return a string representation of the TruthOutput.

        Returns
        -------
        str
            String representation of the TruthOutput.
        """
        return f"TruthOutput({self.output})"
