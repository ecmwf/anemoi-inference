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
from functools import cached_property

LOG = logging.getLogger(__name__)


class Output(ABC):
    """_summary_"""

    def __init__(self, context, output_frequency=None, write_initial_state=None):

        self.context = context
        self.checkpoint = context.checkpoint
        self.reference_date = None

        self._write_step_zero = write_initial_state
        self._output_frequency = output_frequency

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def step(self, state):
        return state["date"] - self.reference_date

    def write_initial_state(self, state):
        self._init(state)
        if self.write_step_zero:
            return self.write_initial_step(state)

    def write_state(self, state):
        self._init(state)

        step = self.step(state)
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        return self.write_step(state)

    def _init(self, state):
        if self.reference_date is not None:
            return

        self.reference_date = state["date"]

        self.open(state)

    def write_initial_step(self, state):
        """This method should not be called directly
        call `write_initial_state` instead.
        """
        reduced_state = self.reduce(state)
        self.write_step(reduced_state)

    @abstractmethod
    def write_step(self, state):
        """This method should not be called directly
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

    @cached_property
    def write_step_zero(self):
        if self._write_step_zero is not None:
            return self._write_step_zero

        return self.context.write_initial_state

    @cached_property
    def output_frequency(self):
        from anemoi.utils.dates import as_timedelta

        if self._output_frequency is not None:
            return as_timedelta(self._output_frequency)

        if self.context.output_frequency is not None:
            return as_timedelta(self.context.output_frequency)

        return None

    def print_summary(self, depth=0):
        LOG.info(
            "%s%s: output_frequency=%s write_initial_state=%s",
            " " * depth,
            self,
            self.output_frequency,
            self.write_step_zero,
        )


class ForwardOutput(Output):
    """
    Subclass of Output that forwards calls to other outputs
    Subclass from that class to implement the desired behaviour of `output_frequency`
    which should only apply to leaves
    """

    def __init__(self, context, output_frequency=None, write_initial_state=None):
        super().__init__(context, output_frequency=None, write_initial_state=write_initial_state)
        if self.context.output_frequency is not None:
            LOG.warning("output_frequency is ignored for '%s'", self.__class__.__name__)

    @cached_property
    def output_frequency(self):
        return None
