# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from abc import ABC
from abc import abstractmethod

LOG = logging.getLogger(__name__)


class Output(ABC):
    """_summary_"""

    def __init__(self, context, output_frequency=None, write_initial_step=False):
        from anemoi.utils.dates import as_timedelta

        self.context = context
        self.checkpoint = context.checkpoint
        self.reference_date = None

        self.write_step_zero = write_initial_step and context.write_initial_step

        self.output_frequency = output_frequency or context.output_frequency
        if self.output_frequency is not None:
            self.output_frequency = as_timedelta(self.output_frequency)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def write_initial_state(self, state):
        self._init(state)
        if self.write_step_zero:
            return self.write_initial_step(state, state["date"] - self.reference_date)

    def write_state(self, state):
        self._init(state)

        step = state["date"] - self.reference_date
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        return self.write_step(state, step)

    def _init(self, state):
        if self.reference_date is not None:
            return

        self.reference_date = state["date"]

        self.open(state)

    def write_initial_step(self, state, step):
        """This method should not be called directly
        call `write_initial_state` instead.
        """
        reduced_state = self.reduce(state)
        self.write_step(reduced_state, step)

    @abstractmethod
    def write_step(self, state, step):
        """This method should be be called directly
        call `write_state` instead.
        """
        pass

    def reduce(self, state):
        """Creates new state which is projection of original state on the last step in the multi-steps dimension."""
        reduced_state = state.copy()
        reduced_state["fields"] = {}
        for field, values in state["fields"].items():
            reduced_state["fields"][field] = values[-1, :]
        return reduced_state

    def open(self, state):
        # Override this method when initialisation is needed
        pass

    def close(self):
        pass
