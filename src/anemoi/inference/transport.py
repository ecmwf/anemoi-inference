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


class Coupling:
    """_summary_"""

    def __init__(self, source, sidx, target, tidx):
        self.source = source
        self.sidx = sidx
        self.target = target
        self.tidx = tidx

    def __str__(self):
        return f"{self.source}:{self.sidx}->{self.target}:{self.tidx}"


class CouplingSend(Coupling):
    """_summary_"""

    def apply(self, task, transport, tensor, tag):
        transport.send_array(task, tensor[self.sidx], self.target, tag)


class CouplingRecv(Coupling):
    """_summary_"""

    def apply(self, task, transport, tensor, tag):
        transport.receive_array(task, tensor[self.tidx], self.source, tag)


class Transport(ABC):
    """_summary_"""

    def __init__(self, couplings, rpcs, tasks):
        self._couplings = couplings
        self.rpcs = rpcs
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def couplings(self, task):

        couplings = []
        for coupling in self._couplings:
            source, target = coupling.split("->")
            source, sidx = source.strip().split(":")
            target, tidx = target.strip().split(":")

            if task.name == source:
                couplings.append(
                    CouplingSend(
                        self.tasks[source],
                        int(sidx),
                        self.tasks[target],
                        int(tidx),
                    )
                )
            if task.name == target:
                couplings.append(
                    CouplingRecv(
                        self.tasks[source],
                        int(sidx),
                        self.tasks[target],
                        int(tidx),
                    )
                )

        return couplings

    # @abstractmethod
    # def send_array(self, data, destination):
    #     """_summary_"""
    #     pass

    # @abstractmethod
    # def receive_array(self, source):
    #     """_summary_"""
    #     pass
