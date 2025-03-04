# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd
from anemoi.transform.filters import filter_registry

from ..processor import Processor

LOG = logging.getLogger(__name__)


class ForwardTransformFilter(Processor):

    def __init__(self, context, filter, **kwargs):
        super().__init__(context)
        self.filter = filter_registry.create(filter, **kwargs)

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:
        return self.filter.forward(fields)

    def patch_data_request(self, data_request):
        return self.filter.patch_data_request(data_request)
