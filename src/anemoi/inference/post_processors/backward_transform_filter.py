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

        fields = self.filter.backward(wrap_state(state))

        return unwrap_state(fields, state, namer=self.context.checkpoint.default_namer())
