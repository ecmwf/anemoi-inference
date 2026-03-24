# (C) Copyright 2024-2025 Anemoi contributors.
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

from anemoi.transform.filters import filter_registry

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.metadata import Metadata
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

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        filter: str | dict[str, Any] | None = None,
        skip_initial_state: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the BackwardTransformFilter.

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

        Examples
        --------
        To initialize a BackwardTransformFilter with a filter name:
        >>> filter_processor = BackwardTransformFilter(context, filter="my_filter")
        To initialize a BackwardTransformFilter with a filter configuration:
        >>> filter_config = {"my_filter": {"param1": value1, "param2": value2}}
        >>> filter_processor = BackwardTransformFilter(context, filter_config)
        """
        super().__init__(context, metadata)
        self.skip_initial_state = skip_initial_state

        if filter is None:
            filter = kwargs
            kwargs = {}
        self.filter = filter_registry.from_config(filter, **kwargs)

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

        fields = self.filter.backward(wrap_state(state))

        return unwrap_state(fields, state, namer=self.metadata.default_namer())

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
    """Apply a transform forward as a post-processor."""

    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.filter = self.filter.reverse()

    def __repr__(self) -> str:
        """Return a string representation of the ForwardTransformFilter object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"ForwardTransformFilter(filter={self.filter})"
