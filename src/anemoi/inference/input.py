# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

from anemoi.inference.pre_processors import create_pre_processor
from anemoi.inference.processor import Processor
from anemoi.inference.types import Date
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context

LOG = logging.getLogger(__name__)

# TODO: only one method is need: `load_data`.
# The other methods can be implemenneted concreetly
# using the `load_data` method.


class Input(ABC):
    """Abstract base class for input handling."""

    trace_name = "????"  # Override in subclass

    def __init__(
        self,
        context: "Context",
        *,
        variables: list[str] | None = None,
        pre_processors: list[ProcessorConfig] | None = None,
        purpose: str | None = None,
    ) -> None:
        """Initialise the Input object.

        Parameters
        ----------
        context : Context
            The context for the input.
        variables : list of str or None
            List of variable names to be handled by the input, or None for all available variables.
        pre_processors : list of ProcessorConfig or None, optional
            List of pre-processors to apply to the input. If None, no pre-processing is performed.
        purpose : str or None, optional
            The purpose of the input (e.g., 'forcings', 'constants'). Used for debugging and logging.
        """
        self.context = context
        self.checkpoint = context.checkpoint
        self._pre_processor_confs = pre_processors or []

        if variables is None:
            variables = self.context.variables.default_input_variables()  # type: ignore

        assert isinstance(variables, list), "variables must be a list of strings"
        self.variables = variables
        self.purpose = purpose

    @cached_property
    def pre_processors(self) -> list[Processor]:
        """Return pre-processors."""

        processors = []
        # inner-level pre-processors
        for processor in self._pre_processor_confs:
            processors.append(create_pre_processor(self.context, processor))

        # top-level pre-processors
        if hasattr(self.context, "pre_processors"):
            processors.extend(self.context.pre_processors)

        LOG.info("Pre-processors: %s", processors)
        return processors

    def pre_process(self, x: Any) -> Any:
        """Run pre-processors.

        Parameters
        ----------
        x : Any
            input to pre-process

        Return
        ------
        Any
            Pre-processed input
        """
        for processor in self.pre_processors:
            LOG.info("Processing with %s", processor)
            x = processor.process(x)
        return x

    def __repr__(self) -> str:
        """Return a string representation of the Input object.

        Returns
        -------
        str
            The string representation of the Input object.
        """
        if self.purpose is None:
            return f"{self.__class__.__name__}(variables={self.variables})"
        else:
            return f"{self.__class__.__name__}({self.purpose})"

    @abstractmethod
    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        """Create the input state dictionary.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        State
            The input state dictionary.
        """
        pass

    @abstractmethod
    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load forcings (constant and dynamic).

        Parameters
        ----------
        dates : List[Date]
            The list of dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        State
            The updated state with the loaded forcings.
        """
        pass

    def input_variables(self) -> list[str]:
        """Return the list of input variables.

        Returns
        -------
        List[str]
            The list of input variables.
        """
        return list(self.checkpoint.variable_to_input_tensor_index.keys())

    def patch_data_request(self, request: Any) -> Any:
        """Patch the data request.

        Uses both the context and input preprocessors.

        Parameters
        ----------
        request : Any
            The data request.

        Returns
        -------
        Any
            The patched data request.
        """
        request = self.context.patch_data_request(request)
        for p in self.pre_processors:
            request = p.patch_data_request(request)

        return request

    def set_private_attributes(self, state: State, value: Any) -> None:
        """Provide a way to a subclass to set private attributes in the state
        dictionary, that may be needed by the output object.

        Parameters
        ----------
        state : State
            The state dictionary.
        value : Any
            The value to set.
        """
        pass
