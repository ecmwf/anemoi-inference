# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from abc import ABC
from abc import abstractmethod


class Processor(ABC):
    """_summary_"""

    def __init__(self, context):
        self.context = context
        self.checkpoint = context.checkpoint

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def process(self, state):
        pass

    def patch_data_request(self, data_request):
        """Override if a processor needs to patch the data request (e.g. mars or cds)"""
        return data_request
