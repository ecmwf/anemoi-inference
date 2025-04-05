# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from typing import List
from typing import Optional

from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context

LOG = logging.getLogger(__name__)


class Output(ABC):
    """Abstract base class for output mechanisms."""

    def __init__(
        self,
        context: "Context",
        variables: Optional[List[str]] = None,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ):
        """Initialize the Output object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """
        self.context = context
        self.checkpoint = context.checkpoint
        self.reference_date = None

        self._write_step_zero = write_initial_state
        self._output_frequency = output_frequency
        self._variables = variables

        self.typed_variables = self.checkpoint.typed_variables.copy()
        self.typed_variables.update(self.context.typed_variables)

    def __repr__(self) -> str:
        """Return a string representation of the Output object.

        Returns
        -------
        str
            String representation of the Output object.
        """
        return f"{self.__class__.__name__}()"

    def write_initial_state(self, state: State) -> None:
        """Write the initial state.

        Parameters
        ----------
        state : State
            The initial state to write.
        """
        state.setdefault("step", datetime.timedelta(0))
        if self.write_step_zero:
            self.write_step(state)

    def write_state(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        self.open(state)

        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        return self.write_step(state)

    @classmethod
    def reduce(cls, state: State) -> State:
        """Create a new state which is a projection of the original state on the last step in the multi-steps dimension.

        Parameters
        ----------
        state : State
            The original state.

        Returns
        -------
        State
            The reduced state.
        """
        reduced_state = state.copy()
        reduced_state["fields"] = {}
        for field, values in state["fields"].items():
            if len(values.shape) == 2:
                reduced_state["fields"][field] = values[-1, :]
            else:
                reduced_state["fields"][field] = values
        return reduced_state

    # @classmethod
    # def update_typed_variables(cls, typed_variables: List[Any], state: State) -> List[Any]:
    #     """Update the typed variables.

    #     Parameters
    #     ----------
    #     typed_variables : list of Any
    #         The list of typed variables.
    #     state : State
    #         The state object.

    #     Returns
    #     -------
    #     list of Any
    #         The updated list of typed variables.
    #     """
    #     # Update the typed variables with the current state

    #     typed_variables = typed_variables.copy()

    #     for name in state["fields"]:
    #         if name not in typed_variables:
    #             LOG.warning("Variable `%s` not found in typed variables, assuming result of a post-processor", name)

    #             typed_variables[name] = PostProcessedVariable(name=name, data={})

    #     return typed_variables

    def open(self, state: State) -> None:
        """Open the output for writing.

        Parameters
        ----------
        state : State
            The state to open.
        """
        # Override this method when initialisation is needed
        pass

    def close(self) -> None:
        """Close the output."""
        pass

    @abstractmethod
    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        pass

    @cached_property
    def write_step_zero(self) -> bool:
        """Determine whether to write the initial state."""
        if self._write_step_zero is not None:
            return self._write_step_zero

        return self.context.write_initial_state

    @cached_property
    def output_frequency(self) -> Optional[datetime.timedelta]:
        """Get the output frequency."""
        from anemoi.utils.dates import as_timedelta

        if self._output_frequency is not None:
            return as_timedelta(self._output_frequency)

        if self.context.output_frequency is not None:
            return as_timedelta(self.context.output_frequency)

        return None

    def print_summary(self, depth: int = 0) -> None:
        """Print a summary of the output configuration.

        Parameters
        ----------
        depth : int, optional
            The indentation depth for the summary, by default 0.
        """
        LOG.info(
            "%s%s: output_frequency=%s write_initial_state=%s",
            " " * depth,
            self,
            self.output_frequency,
            self.write_step_zero,
        )


class ForwardOutput(Output):
    """Subclass of Output that forwards calls to other outputs.

    Subclass from this class to implement the desired behaviour of `output_frequency`
    which should only apply to leaves.

    Parameters
    ----------
    context : Context
        The context in which the output operates.
    output_frequency : Optional[int], optional
        The frequency at which to output states, by default None.
    write_initial_state : Optional[bool], optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self, context: "Context", output_frequency: Optional[int] = None, write_initial_state: Optional[bool] = None
    ):
        """Initialize the ForwardOutput object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """
        super().__init__(context, output_frequency=None, write_initial_state=write_initial_state)
        if self.context.output_frequency is not None:
            LOG.warning("output_frequency is ignored for '%s'", self.__class__.__name__)

    @cached_property
    def output_frequency(self) -> Optional[datetime.timedelta]:
        """Get the output frequency."""
        return None
