# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from abc import ABC
from abc import abstractmethod


class Output(ABC):
    """_summary_"""

    def __init__(self, context):
        self.context = context
        self.checkpoint = context.checkpoint

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def write_initial_state(self, state):
        pass

    @abstractmethod
    def write_state(self, state):
        pass

    def close(self):
        pass
