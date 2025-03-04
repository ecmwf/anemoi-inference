# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

from ..output import ForwardOutput
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("tee")
class TeeOutput(ForwardOutput):
    """TeeOutput class to manage multiple outputs.

    Attributes
    ----------
    outputs : list
        List of output objects.
    """

    def __init__(
        self,
        context: object,
        *args: tuple,
        outputs: list | tuple | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        **kwargs: dict,
    ):
        """Initialize the TeeOutput.

        Parameters
        ----------
        context : object
            The context object.
        *args : tuple
            Additional positional arguments.
        outputs : list or tuple, optional
            List of outputs to be created.
        output_frequency : int, optional
            Frequency of output.
        write_initial_state : bool, optional
            Flag to write the initial state.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)

        if outputs is None:
            outputs = args

        assert isinstance(outputs, (list, tuple)), outputs
        self.outputs = [create_output(context, output) for output in outputs]

    # We override write_initial_state and write_state
    # so users can configures each levels independently
    def write_initial_state(self, state: dict) -> None:
        """Write the initial state to all outputs.

        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        state.setdefault("step", datetime.timedelta(0))
        for output in self.outputs:
            output.write_initial_state(state)

    def write_state(self, state: dict) -> None:
        """Write the state to all outputs.

        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        for output in self.outputs:
            output.write_state(state)

    def write_step(self, state: dict) -> None:
        """Raise NotImplementedError as TeeOutput does not support write_step.

        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        raise NotImplementedError("TeeOutput does not support write_step")

    def open(self, state: dict) -> None:
        """Open all outputs.

        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        for output in self.outputs:
            output.open(state)

    def close(self) -> None:
        """Close all outputs."""
        for output in self.outputs:
            output.close()

    def __repr__(self) -> str:
        """Return a string representation of the TeeOutput.

        Returns
        -------
        str
            String representation of the TeeOutput.
        """
        return f"TeeOutput({self.outputs})"

    def print_summary(self, depth: int = 0) -> None:
        """Print the summary of all outputs.

        Parameters
        ----------
        depth : int, optional
            The depth of the summary.
        """
        super().print_summary(depth)
        for output in self.outputs:
            output.print_summary(depth + 1)
