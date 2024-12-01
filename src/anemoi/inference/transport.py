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

LOG = logging.getLogger(__name__)


class Transport(ABC):
    """_summary_"""

    def __init__(self, couplings):
        self._couplings = couplings

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def couplings(self, task_name):
        return [c for c in self._couplings if c.source.name == task_name or c.target.name == task_name]

    # @abstractmethod
    # def send(self, data, destination):
    #     """_summary_"""
    #     pass

    # @abstractmethod
    # def receive(self, source):
    #     """_summary_"""
    #     pass
