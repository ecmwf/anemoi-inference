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

import earthkit.data as ekd
from anemoi.transform.filters import filter_registry

from anemoi.inference.decorators import main_argument

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

    def __init__(self, context: Any, filter: str, **kwargs: Any) -> None:
        """Initialize the ForwardTransformFilter.

        Parameters
        ----------
        context : object
            The context in which the filter is being used.
        filter : str
            The name of the filter to be used.
        **kwargs : dict
            Additional keyword arguments to pass to the filter.
        """
        super().__init__(context)
        self.filter = filter_registry.create(filter, **kwargs)

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Process the given fields using the forward filter.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to be processed.

        Returns
        -------
        ekd.FieldList
            The processed fields.
        """
        return self.filter.forward(fields)

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
