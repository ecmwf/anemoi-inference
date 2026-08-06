# (C) Copyright 2024-2026 Anemoi contributors.
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
from typing import Any

from anemoi.inference.post_processors import create_post_processor
from anemoi.inference.processor import Processor
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context
    from anemoi.inference.metadata import Metadata

LOG = logging.getLogger(__name__)


def _parse_levels(level: Any) -> list:
    """Expand a level specification into a list of concrete levels.

    Accepts a single level, a list of levels, or a MARS-style range string
    such as ``"1/to/50"``, ``"50/to/137"`` or ``"1/to/100/by/3"``.
    """
    if isinstance(level, (list, tuple)):
        return list(level)

    if isinstance(level, str) and "/" in level:
        tokens = [token.strip() for token in level.split("/")]

        if len(tokens) not in (3, 5) or tokens[1].lower() != "to" or (len(tokens) == 5 and tokens[3].lower() != "by"):
            raise ValueError(
                f"Invalid level range {level!r}: expected '<start>/to/<stop>' or '<start>/to/<stop>/by/<step>'"
            )

        # Parse the two range endpoints
        start, stop = int(tokens[0]), int(tokens[2])
        # Parse the 'by' step if given (5 tokens: start/to/stop/by/step), otherwise default to 1.
        step = int(tokens[4]) if len(tokens) == 5 else 1
        if step <= 0:
            raise ValueError(f"Invalid level range {level!r}: 'by' step must be a positive integer")
        # 'by' is always given as a positive number, even for a descending
        # range (e.g. "137/to/50/by/2"), so flip its sign here when start > stop.
        step = step if start <= stop else -step
        # range() is exclusive of its stop value, so extend it by one step in the
        # direction of iteration to make our result inclusive of the given stop level.
        return list(range(start, stop + (1 if step > 0 else -1), step))

    return [level]


class Output(ABC):
    """Abstract base class for output mechanisms."""

    def __init__(
        self,
        context: "Context",
        metadata: "Metadata",
        *,
        variables: list[str | dict] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ):
        """Initialize the Output object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        metadata : Metadata
            Metadata corresponding to the dataset this output is handling.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the output
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """
        self.context = context
        self.metadata = metadata
        self.dataset_name = metadata.dataset_name
        self.reference_date = context.reference_date

        self._post_processor_confs = post_processors or []

        self._write_step_zero = write_initial_state
        self._output_frequency = output_frequency

        self.variables = variables
        if self.variables is not None:
            if not isinstance(self.variables, (list, tuple)):
                self.variables = [self.variables]

        # ``variables`` may mix plain names (e.g. ``"t_137"``) with structured
        # selectors (e.g. ``{"param": "u", "level": "50/to/137"}``). Collect the
        # exact names, and turn each selector into a ``(param, levels)`` pair
        # where ``levels`` is ``None`` to mean "every level of this param".
        self._names: set[str] = set()
        self._selectors: list[tuple[str, set | None]] = []
        for entry in self.variables or []:
            if isinstance(entry, dict):
                level = entry.get("level")
                levels = None if level is None else {str(lvl) for lvl in _parse_levels(level)}
                self._selectors.append((entry["param"], levels))
            else:
                self._names.add(entry)

        self.typed_variables = self.metadata.typed_variables.copy()
        self.typed_variables.update(self.context.typed_variables)

    def skip_variable(self, variable: str) -> bool:
        """Check if a variable should be skipped.

        Parameters
        ----------
        variable : str
            The variable to check.

        Returns
        -------
        bool
            True if the variable should be skipped, False otherwise.
        """
        if self.variables is None:
            return False

        if variable in self._names:
            return False

        typed = self.typed_variables.get(variable)
        if typed is not None:
            level = None if typed.level is None else str(typed.level)
            for param, levels in self._selectors:
                if param == typed.param and (levels is None or level in levels):
                    return False

        return True

    @cached_property
    def post_processors(self) -> list[Processor]:
        """Return post-processors."""

        processors = []

        for processor in self._post_processor_confs:
            processors.append(create_post_processor(self.context, processor, self.metadata))

        return processors

    def post_process(self, state: State) -> State:
        """Apply post processors to the state.

        Parameters
        ----------
        state : State
            The state.

        Returns
        -------
        State
            The processed state.
        """
        for processor in self.post_processors:
            LOG.info("Post processor: %s", processor)
            state = processor.process(state)
        return state

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
            self.write_step(self.post_process(state))

    def write_state(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        return self.write_step(self.post_process(state))

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
            if len(values.shape) > 1:
                reduced_state["fields"][field] = values[-1, :]
            else:
                reduced_state["fields"][field] = values
        return reduced_state

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
    def output_frequency(self) -> datetime.timedelta | None:
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

    """

    def __init__(
        self,
        context: "Context",
        metadata: "Metadata",
        output: Output | Any,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ):
        """Initialise the ForwardOutput object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output : Output | Any
            The output configuration dictionary or an Output instance.
        variables : list, optional
            The list of variables, by default None.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """

        from anemoi.inference.outputs import create_output

        super().__init__(
            context,
            metadata,
            variables=variables,
            post_processors=post_processors,
            output_frequency=None,
            write_initial_state=write_initial_state,
        )
        if not isinstance(output, Output):
            output = create_output(context, output, self.metadata)
        self.output = output

        if self.context.output_frequency is not None:
            LOG.warning("output_frequency is ignored for '%s'", self.__class__.__name__)

    @cached_property
    def output_frequency(self) -> datetime.timedelta | None:
        """Get the output frequency."""
        return None

    def modify_state(self, state: State) -> State:
        """Modify the state before writing.

        Parameters
        ----------
        state : State
            The state to modify.

        Returns
        -------
        State
            The modified state.
        """
        return state

    def open(self, state) -> None:
        """Open the output for writing.
        Parameters
        ----------
        state : State
            The initial state.
        """
        self.output.open(self.modify_state(state))

    def close(self) -> None:
        """Close the output."""

        self.output.close()

    def write_initial_state(self, state: State) -> None:
        """Write the initial step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        state.setdefault("step", datetime.timedelta(0))

        self.output.write_initial_state(self.modify_state(state))

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        self.output.write_state(self.modify_state(state))

    def print_summary(self, depth: int = 0) -> None:
        """Print a summary of the output.

        Parameters
        ----------
        depth : int, optional
            The depth of the summary, by default 0.
        """
        super().print_summary(depth)
        self.output.print_summary(depth + 1)
