# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from abc import ABC, abstractmethod

from .input import Input
LOG = logging.getLogger(__name__)

# TODO: only one method is need: `load_data`.
# The other methods can be implemenneted concreetly
# using the `load_data` method.


class Input_0(Input):
    """_summary_"""

    def __init__(self, context):
        super().__init__(context)
        self.checkpoint_0 = context.checkpoint_0


    def input_0_variables(self):
        """Return the list of input variables"""
        return list(self.checkpoint_0.variable_to_input_tensor_index.keys())

class Input_1(Input):
    """_summary_"""

    def __init__(self, context):
        super().__init__(context)
        self.checkpoint_1 = context.checkpoint_1


    def input_1_variables(self):
        """Return the list of input variables"""
        return list(self.checkpoint_1.variable_to_input_tensor_index.keys())