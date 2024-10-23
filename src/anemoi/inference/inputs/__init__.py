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


class Input(ABC):
    """_summary_"""

    def __init__(self, checkpoint, verbose=True):
        self.checkpoint = checkpoint
        self._verbose = verbose

    @abstractmethod
    def create_input_state(self, *, date=None):
        pass

    def set_private_attributes(self, state):
        """Provide a way to a subclass to set private attributes in the state
        dictionary, that may be needed but the output object.
        """
        pass
