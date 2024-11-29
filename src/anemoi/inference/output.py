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

    def reduce(self, state):
        """Creates new state which is projection of original state on the last step in the multi-steps dimension."""
        reduced_state = state.copy()
        reduced_state["fields"] = {}
        for field, values in state["fields"].items():
            reduced_state["fields"][field] = values[-1, :]
        return reduced_state

    def close(self):
        pass
