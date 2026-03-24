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

from anemoi.inference.decorators import main_argument
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State

from ..processor import Processor
from . import pre_processor_registry

LOG = logging.getLogger(__name__)


@pre_processor_registry.register("forward_transform_filter")
@main_argument("filter")
class ForwardTransformFilter(Processor):
    """A processor that applies a forward transform filter to the given fields.

    This class uses a specified filter from the filter registry to process
    fields and patch data requests.

    Attributes
    ----------
    filter : object
        The filter instance used for processing fields and patching data requests.
    """

    def __init__(
        self, context: Any, metadata: Metadata, *, filter: str | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Initialize the ForwardTransformFilter.

        Parameters
        ----------
        context : object
            The context in which the filter is being used.
        metadata : Metadata
            Metadata corresponding to the dataset this processor is handling.
        filter : str | dict[str, Any] | None, optional
            The name of the filter or a configuration dictionary for the filter, by default None

        Examples
        --------
        To initialize a BackwardTransformFilter with a filter name:
        >>> filter_processor = BackwardTransformFilter(context, filter="my_filter")
        To initialize a BackwardTransformFilter with a filter configuration:
        >>> filter_config = {"my_filter": {"param1": value1, "param2": value2}}
        >>> filter_processor = BackwardTransformFilter(context, filter_config)
        """
        super().__init__(context, metadata)
        if filter is None:
            filter = kwargs
            kwargs = {}
        self.filter = filter_registry.from_config(filter, **kwargs)

    def process(self, state: State) -> State:
        """Process the given fields using the forward filter.

        Parameters
        ----------
        state : State
            The state containing the fields to be processed.

        Returns
        -------
        State
            The processed state.
        """
        state["fields"] = self.filter.forward(state["fields"])
        return state

    def patch_data_request(self, data_request: Any) -> Any:
        """Patch the data request using the filter.

        Parameters
        ----------
        data_request : object
            The data request to be patched.

        Returns
        -------
        object
            The patched data request.
        """
        return self.filter.patch_data_request(data_request)

    def __repr__(self) -> str:
        """Return a string representation of the ForwardTransformFilter object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"ForwardTransformFilter(filter={self.filter})"
