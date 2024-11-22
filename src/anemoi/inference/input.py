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

# TODO: only one method is need: `load_data`.
# The other methods can be implemenneted concreetly
# using the `load_data` method.


class Input(ABC):
    """_summary_"""

    def __init__(self, context):
        self.context = context
        self.checkpoint = context.checkpoint

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def create_input_state(self, *, date=None):
        """Create the input state dictionary."""
        pass

    @abstractmethod
    def load_forcings(self, *, variables, dates):
        """Load forcings (constant and dynamic)."""
        pass

    def input_variables(self):
        """Return the list of input variables"""
        return list(self.checkpoint.variable_to_input_tensor_index.keys())

    def set_private_attributes(self, state, value):
        """Provide a way to a subclass to set private attributes in the state
        dictionary, that may be needed but the output object.
        """
        pass

    def template(self, variable, date, **kwargs):
        """Used for fetching GRIB templates."""
        raise NotImplementedError(f"{self.__class__.__name__}.template() not implemented")
