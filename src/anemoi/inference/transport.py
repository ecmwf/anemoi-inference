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
from typing import Any

from anemoi.utils.logs import enable_logging_name

from anemoi.inference.task import Task
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


class Coupling:
    """Represents a coupling between a source and a target with specific variables."""

    def __init__(self, source: Task, target: Task, variables: list[str]) -> None:
        """Initialize a Coupling instance.

        Parameters
        ----------
        source : Task
            The source of the coupling.
        target : Task
            The target of the coupling.
        variables : List[str]
            The variables involved in the coupling.
        """
        self.source = source
        self.target = target
        self.variables: list[str] = variables

    def __repr__(self) -> str:
        """Return a string representation of the coupling.

        Returns
        -------
        str
            The string representation of the coupling.
        """
        return f"{self.source}->{self.target}(variables={self.variables})"


class CouplingSend(Coupling):
    """Represents a coupling send operation."""

    def apply(
        self,
        task: Task,
        transport: "Transport",
        *,
        input_state: State,
        output_state: State,
        constants: State,
        tag: int,
    ) -> None:
        """Apply the coupling send operation.

        Parameters
        ----------
        task : Task
            The task to apply the coupling to.
        transport : Transport
            The transport instance to use.
        input_state : State
            The input state dictionary.
        output_state : State
            The output state dictionary.
        constants : State
            The constants dictionary.
        tag : int
            The tag for the operation.
        """
        transport.send_state(
            task,
            self.target,
            input_state=input_state,
            variables=self.variables,
            constants=constants,
            tag=tag,
        )


class CouplingRecv(Coupling):
    """Represents a coupling receive operation."""

    def apply(
        self,
        task: Task,
        transport: "Transport",
        *,
        input_state: State,
        output_state: State,
        constants: State,
        tag: int,
    ) -> None:
        """Apply the coupling receive operation.

        Parameters
        ----------
        task : Any
            The task to apply the coupling to.
        transport : Transport
            The transport instance to use.
        input_state : State
            The input state dictionary.
        output_state : State
            The output state dictionary.
        constants : State
            The constants dictionary.
        tag : str
            The tag for the operation.
        """
        transport.receive_state(
            task,
            self.source,
            output_state=output_state,
            variables=self.variables,
            tag=tag,
        )


class Transport(ABC):
    """Abstract base class for transport mechanisms."""

    def __init__(self, couplings: list[dict[str, list[str]]], tasks: dict[str, Task]) -> None:
        """Initialize a Transport instance.

        Parameters
        ----------
        couplings : List[Dict[str,str]]
            The list of dict representing the couplings.
        tasks : Dict[str, Task]
            The dictionary of tasks.
        """
        enable_logging_name("main")
        self._couplings = couplings
        self.tasks: dict[str, Task] = tasks

    @abstractmethod
    def send(self, sender: Task, target: Task, state: State, tag: int) -> None:
        """Send the state from the sender to the target.

        Parameters
        ----------
        sender : Task
            The sender of the state.
        target : Task
            The target
        state : State
            The state dictionary to send.
        tag : int
            The tag for the operation.
        """
        pass

    @abstractmethod
    def receive(self, receiver: Task, source: Task, tag: int) -> State:
        """Receive the state from the source to the receiver.

        Parameters
        ----------
        receiver : Task
            The receiver of the state.
        source : Task
            The source of the state.
        tag : int
            The tag for the operation.

        Returns
        -------
        State
            The received state dictionary.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the transport.

        Returns
        -------
        str
            The string representation of the transport.
        """
        return f"{self.__class__.__name__}()"

    def couplings(self, task: Task) -> list[Coupling]:
        """Get the couplings for a given task.

        Parameters
        ----------
        task : Task
            The task to get the couplings for.

        Returns
        -------
        List[Coupling]
            The list of couplings for the task.
        """
        couplings: list[Coupling] = []
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
        sender: Task,
        target: Task,
        *,
        input_state: State,
        variables: list[str],
        constants: dict[str, Any],
        tag: int,
    ) -> None:
        """Send the state from the sender to the target.

        Parameters
        ----------
        sender : Any
            The sender of the state.
        target : Any
            The target of the state.
        input_state : Dict[str, Any]
            The input state dictionary.
        variables : List[str]
            The list of variables to send.
        constants : Dict[str, Any]
            The constants dictionary.
        tag : int
            The tag for the operation.
        """
        assert sender.name != target.name, f"Cannot send to self {sender}"

        fields: dict[str, Any] = input_state["fields"]

        LOG.info(f"{sender}: sending to {target} {variables} {input_state['date']}")
        LOG.info("State fields: %s", list(fields.keys()))

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

        state: State = input_state.copy()
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
        receiver: Task,
        source: Task,
        *,
        output_state: State,
        variables: list[str],
        tag: int,
    ) -> None:
        """Receive the state from the source to the receiver.

        Parameters
        ----------
        receiver : Any
            The receiver of the state.
        source : Any
            The source of the state.
        output_state : Dict[str, Any]
            The output state dictionary.
        variables : List[str]
            The list of variables to receive.
        tag : int
            The tag for the operation.
        """
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"

        state: State = self.receive(receiver, source, tag)

        assert isinstance(state, dict), state
        assert "fields" in state
        assert isinstance(state["fields"], dict), f"Expected dict got {type(state['fields'])}"

        output_state.setdefault("fields", {})

        fields_in: dict[str, Any] = state["fields"]
        fields_out: dict[str, Any] = output_state["fields"]

        for v in variables:
            if v in fields_out:
                raise ValueError(f"Variable {v} already in output state")

            if v not in fields_in:
                raise ValueError(f"Variable {v} not in input state")

            fields_out[v] = fields_in[v]

            # assert len(fields_out[v].shape) == 1, f"Expected  got {fields_out[v].shape}"
