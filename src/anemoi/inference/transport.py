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
from typing import Any
from typing import Dict
from typing import List

from anemoi.utils.logs import enable_logging_name

LOG = logging.getLogger(__name__)


class Coupling:
    def __init__(self, source: Any, target: Any, variables: List[str]) -> None:
        self.source = source
        self.target = target
        self.variables = variables

    def __str__(self) -> str:
        return f"{self.source}->{self.target}"


class CouplingSend(Coupling):
    def apply(
        self,
        task: Any,
        transport: Any,
        *,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        constants: Dict[str, Any],
        tag: str,
    ) -> None:
        transport.send_state(
            task,
            self.target,
            input_state=input_state,
            variables=self.variables,
            constants=constants,
            tag=tag,
        )


class CouplingRecv(Coupling):
    def apply(
        self,
        task: Any,
        transport: Any,
        *,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
        constants: Dict[str, Any],
        tag: str,
    ) -> None:
        transport.receive_state(
            task,
            self.source,
            output_state=output_state,
            variables=self.variables,
            tag=tag,
        )


class Transport(ABC):
    def __init__(self, couplings: List[Dict[str, List[str]]], tasks: Dict[str, Any]) -> None:
        enable_logging_name("main")
        self._couplings = couplings
        self.tasks = tasks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def couplings(self, task: Any) -> List[Coupling]:
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

    def send_state(
        self,
        sender: Any,
        target: Any,
        *,
        input_state: Dict[str, Any],
        variables: List[str],
        constants: Dict[str, Any],
        tag: str,
    ) -> None:
        assert sender.name != target.name, f"Cannot send to self {sender}"

        fields = input_state["fields"]

        LOG.info(f"{sender}: sending to {target} {variables} {input_state['date']}")

        fields = {v: fields[v] for v in variables if v in fields}

        for v in variables:
            if v not in fields:
                # Check in the constants
                if v in constants:
                    LOG.warning(f"{sender}: {v} not in fields, using the value from constants")
                    fields[v] = constants[v]
                else:
                    raise ValueError(f"{sender}: Variable {v} not in fields or constants")

        for f, v in fields.items():
            assert len(v.shape) == 1, f"Expected  got {v.shape}"

        state = input_state.copy()
        state["fields"] = fields

        # Don't send unnecessary data
        state["latitudes"] = None
        state["longitudes"] = None
        for s in list(state.keys()):
            if s.startswith("_"):
                del state[s]

        self.send(sender, target, state, tag)

    def receive_state(
        self,
        receiver: Any,
        source: Any,
        *,
        output_state: Dict[str, Any],
        variables: List[str],
        tag: str,
    ) -> None:
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"

        state = self.receive(receiver, source, tag)

        assert isinstance(state, dict)
        assert "fields" in state
        assert isinstance(state["fields"], dict), f"Expected dict got {type(state['fields'])}"

        output_state.setdefault("fields", {})

        fields_in = state["fields"]
        fields_out = output_state["fields"]

        for v in variables:
            if v in fields_out:
                raise ValueError(f"Variable {v} already in output state")

            if v not in fields_in:
                raise ValueError(f"Variable {v} not in input state")

            fields_out[v] = fields_in[v]

            assert len(fields_out[v].shape) == 1, f"Expected  got {fields_out[v].shape}"
