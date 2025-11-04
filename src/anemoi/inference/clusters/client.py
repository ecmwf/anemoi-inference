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
from typing import Protocol

from anemoi.inference.lazy import torch

LOG = logging.getLogger(__name__)


class ClusterClientProtocol(Protocol):
    @classmethod
    def used(cls) -> bool:
        """Check if this client is valid in the current environment."""
        ...


@dataclass
class ComputeClient:
    world_size: int

    local_rank: int
    global_rank: int

    master_addr: str
    master_port: int

    process_group: "torch.distributed.ProcessGroup | None"

    @property
    def is_master(self) -> bool:
        """Return True if the current process is the master process."""
        return self.global_rank == 0


class ComputeClientFactory(ABC):
    """Abstract factory class for compute client creation."""

    def create_client(self) -> ComputeClient:
        """Create and return a ComputeClient instance."""
        return ComputeClient(
            process_group=self.create_model_comm_group(),
            world_size=self.world_size,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            master_addr=self.master_addr,
            master_port=self.master_port,
        )

    @classmethod
    @abstractmethod
    def used(cls) -> bool:
        """Check if this client is valid in the current environment."""
        raise NotImplementedError

    @property
    def init_method(self) -> str:
        """Return the initialisation method string for distributed computing."""
        return f"tcp://{self.master_addr}:{self.master_port}"

    @property
    def backend(self) -> str:
        """Return the backend for distributed computing."""
        return "nccl" if torch.cuda.is_available() else "gloo"  # type: ignore

    def create_model_comm_group(self) -> "torch.distributed.ProcessGroup | None":
        """Create the communication group for model parallelism."""
        if self.world_size <= 1:
            return None

        LOG.debug("Creating model communication group for parallel inference")
        group = torch.distributed.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            timeout=datetime.timedelta(minutes=3),
            world_size=self.world_size,
            rank=self.global_rank,
        )

        # Create a new process group for model communication
        group = torch.distributed.new_group(
            ranks=list(range(self.world_size)),
        )
        LOG.info("Model communication group created")

        return group

    @property
    def is_master(self) -> bool:
        """Return True if the current process is the master process."""
        return self.global_rank == 0

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        raise NotImplementedError

    @property
    def device_index(self) -> int:
        """Return the device index for the current process, defaults to local rank."""
        return self.local_rank

    @property
    @abstractmethod
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Return the total number of processes in the cluster."""
        raise NotImplementedError

    @property
    @abstractmethod
    def master_addr(self) -> str:
        """Return the master address."""
        raise NotImplementedError

    @property
    @abstractmethod
    def master_port(self) -> int:
        """Return the master port."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(world_size={self.world_size}, "
            f"global_rank={self.global_rank}, local_rank={self.local_rank}, "
            f"master_addr='{self.master_addr}', master_port={self.master_port})"
        )
