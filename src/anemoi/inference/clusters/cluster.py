# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import NamedTuple

from anemoi.inference.context import Context
from anemoi.inference.lazy import torch

ADDRESS = NamedTuple("Address", [("host", str), ("port", int)])
LOG = logging.getLogger(__name__)


@dataclass
class CommsGroup:
    """Data class for communication group initialisation parameters."""

    world_size: int
    global_rank: int

    init_kwargs: dict[str, Any]


class Cluster(ABC):
    """Abstract base class for cluster and parallel environment handling."""

    _model_comm_group: torch.distributed.ProcessGroup | None = None  # type: ignore

    def __init__(self, context: Context) -> None:
        """Cluster class for parallel inference

        Parameters
        ----------
        context : Context
            Runner context
        kwargs : Any
            Additional keyword arguments
        """
        self.context = context

    def configure(self, *, pid: int) -> None:
        """Configure the cluster with additional parameters.

        Parameters
        ----------
        pid : int
            The process ID.
            Used to set local rank in some cluster implementations.
        """
        pass

    @classmethod
    @abstractmethod
    def used(cls) -> bool:
        """Check if this cluster is valid in the current environment."""
        raise NotImplementedError("Subclasses must implement this method.")

    def spawn(self, fn: Any, *args: Any) -> None:
        """Spawn processes in the cluster environment.

        Should be overridden by subclasses if specific spawning logic is required.
        By default, this method does nothing.

        Parameters
        ----------
        fn : Any
            The function to run in each process.
        args : tuple[Any, ...]
            The arguments to pass to the function.
        """
        pass

    @property
    def init_method(self) -> str:
        """Return the initialisation method string for distributed computing."""
        return f"tcp://{self.master_addr}:{self.master_port}"

    @property
    def backend(self) -> str:
        """Return the backend for distributed computing."""
        return "nccl" if self.context.device.type == "cuda" else "gloo"  # type: ignore

    @property
    def comm_group_init(self) -> CommsGroup:
        """Return the communication group initialisation parameters."""
        return CommsGroup(
            world_size=self.world_size,
            global_rank=self.global_rank,
            init_kwargs={
                "backend": self.backend,
                "init_method": self.init_method,
                "timeout": datetime.timedelta(minutes=3),
                "world_size": self.world_size,
                "rank": self.global_rank,
            },
        )

    def teardown(self) -> None:
        """Tear down the cluster environment."""
        pass

    def __del__(self) -> None:
        self.teardown()

    @property
    def is_master(self) -> bool:
        """Return True if the current process is the master process."""
        return self.global_rank == 0

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Return the total number of processes in the cluster."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def master_addr(self) -> str:
        """Return the master address."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def master_port(self) -> int:
        """Return the master port."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def address(self) -> ADDRESS:
        """Return the master address and port as an ADDRESS named tuple."""
        return ADDRESS(self.master_addr, self.master_port)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(world_size={self.world_size}, "
            f"global_rank={self.global_rank}, local_rank={self.local_rank}, "
            f"master_addr='{self.master_addr}', master_port={self.master_port})"
        )
