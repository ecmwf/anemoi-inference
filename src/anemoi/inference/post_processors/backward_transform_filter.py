# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.transform.filters import filter_registry
from earthkit.data import FieldList

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry
from .earthkit_state import unwrap_state
from .earthkit_state import wrap_state

LOG = logging.getLogger(__name__)


@post_processor_registry.register("backward_transform_filter")
@main_argument("filter")
class BackwardTransformFilter(Processor):
    """A processor that applies a backward transform filter to a given state.

    This class uses a specified filter from the filter registry to process
    the state by applying a backward transformation.

    Attributes
    ----------
    filter : Any
        The filter instance used for processing the state.
    """

    def __init__(self, context: Context, filter: str, **kwargs: Any) -> None:
        """Initialize the BackwardTransformFilter.

        Parameters
        ----------
        context : Context
            The context for the filter.
        filter : str
            The name of the filter to be used.
        **kwargs : Any
            Additional keyword arguments for the filter.
        """
        super().__init__(context)
        self.filter: Any = filter_registry.create(filter, **kwargs)

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

        fields = self._exec_filter(wrap_state(state))

        return unwrap_state(fields, state, namer=self.context.checkpoint.default_namer())

    def _exec_filter(self, state: FieldList) -> FieldList:
        return self.filter.backward(state)

    def __repr__(self) -> str:
        """Return a string representation of the BackwardTransformFilter object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"BackwardTransformFilter(filter={self.filter})"


@post_processor_registry.register("forward_transform_filter")
class ForwardTransformFilter(BackwardTransformFilter):
    """Apply a transform forward or reversed backward as a post-processor."""

    def __init__(self, *args: Any, use_forward: bool = False, **kwargs: Any) -> None:
        """Initialize the ForwardTransformFilter.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the parent's __init__.
        use_forward : bool
            Whether to use the forward method of the transform filter. If False, will
            use the backward transform with the .reverse() method applied to the filter.
            Defaults to False.
        **kwargs : Any
            Additional arguments to pass to the parent's __init__.
        """
        super().__init__(*args, **kwargs)

        self.use_forward = use_forward

        if not self.use_forward:
            self.filter = self.filter.reverse()

    def _exec_filter(self, state: FieldList) -> FieldList:
        """Process the given state using the forward transform filter if use_forward=True,
        otherwise uses the backward transform filter with the filter reversed.
        """

        if self.use_forward:
            return self.filter.forward(state)

        return self.filter.backward(state)

    def __repr__(self) -> str:
        """Return a string representation of the ForwardTransformFilter object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"ForwardTransformFilter(filter={self.filter}, use_forward={self.use_forward})"
