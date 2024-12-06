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

    def __init__(self, source, target, variables):
        self.source = source
        self.target = target
        self.variables = variables

    def __str__(self):
        return f"{self.source}->{self.target}"


class CouplingSend(Coupling):
    """_summary_"""

    def apply(self, task, transport, *, input_state, output_state, constants):
        transport.send_state(
            task,
            self.target,
            input_state=input_state,
            variables=self.variables,
            constants=constants,
        )


class CouplingRecv(Coupling):
    """_summary_"""

    def apply(self, task, transport, *, input_state, output_state, constants):
        transport.receive_state(
            task,
            self.source,
            output_state=output_state,
            variables=self.variables,
        )


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
            assert isinstance(coupling, dict)
            assert len(coupling) == 1
            k, variables = list(coupling.items())[0]
            source, target = k.split("->")
            source = source.strip()
            target = target.strip()

            if task.name == source:
                couplings.append(
                    CouplingSend(
                        self.tasks[source],
                        self.tasks[target],
                        variables,
                    )
                )
            if task.name == target:
                couplings.append(
                    CouplingRecv(
                        self.tasks[source],
                        self.tasks[target],
                        variables,
                    )
                )

        return couplings

    # @abstractmethod
    # def send_state(self, data, destination):
    #     """_summary_"""
    #     pass

    # @abstractmethod
    # def receive_state(self, source):
    #     """_summary_"""
    #     pass
