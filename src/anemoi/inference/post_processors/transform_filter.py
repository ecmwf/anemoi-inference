# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from datetime import timedelta
from typing import Any

from anemoi.transform.filters import filter_registry
from earthkit.data import FieldList

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry
from .earthkit_state import unwrap_state
from .earthkit_state import wrap_state

LOG = logging.getLogger(__name__)


@main_argument("filter")
class TransformFilter(Processor, ABC):
    """A processor that applies a transform filter to a given state.

    This class uses a specified filter from the filter registry to process
    the state by applying a forward or backward transformation.

    Attributes
    ----------
    filter : Any
        The filter instance used for processing the state.
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        filter: str | dict[str, Any] | None = None,
        skip_initial_state: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the TransformFilter.

        Parameters
        ----------
        context : Context
            The context in which the filter is being used.
        metadata : Metadata
            Metadata corresponding to the dataset this processor is handling.
        filter : str | dict[str, Any] | None, optional
            The name of the filter or a configuration dictionary for the filter, by default None
        skip_initial_state : bool, optional
            Whether to skip processing the initial state, by default False
        """
        super().__init__(context, metadata)
        self.skip_initial_state = skip_initial_state

        if filter is None:
            filter = kwargs
            kwargs = {}
        self.filter = filter_registry.from_config(filter, **kwargs)

    @abstractmethod
    def _exec_filter(self, state: FieldList) -> FieldList:
        """Process the given state using the filter.

        Parameters
        ----------
        state : FieldList
            The state to be processed.

        Returns
        -------
        FieldList
            The processed state.
        """
        raise NotImplementedError("Subclasses must implement the _exec_filter method.")

    def process(self, state: State) -> State:
        """Process the given state using the backward transform filter.

        Parameters
        ----------
        state : State
            The state to be processed.

        Returns
        -------
        State
            The processed state.
        """
        if self.skip_initial_state and ("step" not in state or state["step"] == timedelta(0)):
            return state

        fields = self._exec_filter(wrap_state(state, self.metadata.typed_variables))

        return unwrap_state(fields, state, namer=self.metadata.default_namer())


@post_processor_registry.register("backward_transform_filter")
class BackwardTransformFilter(TransformFilter):
    """A processor that applies a backward transform filter to a given state.

    This class uses a specified filter from the filter registry to process
    the state by applying a backward transformation.
    """

    def _exec_filter(self, state: FieldList) -> FieldList:
        return self.filter.backward(state)

    def __repr__(self) -> str:
        return f"BackwardTransformFilter(filter={self.filter})"


@post_processor_registry.register("forward_transform_filter")
class ForwardTransformFilter(BackwardTransformFilter):
    """A processor that applies a forward transform filter to a given state.

    This class uses a specified filter from the filter registry to process
    the state by applying a forward transformation.
    """

    def _exec_filter(self, state: FieldList) -> FieldList:
        return self.filter.forward(state)

    def __repr__(self) -> str:
        return f"ForwardTransformFilter(filter={self.filter})"
