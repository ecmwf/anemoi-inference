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
import os
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import NamedTuple

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.lazy import torch

ADDRESS = NamedTuple("Address", [("host", str), ("port", int)])
LOG = logging.getLogger(__name__)


class Cluster(ABC):
    """Abstract base class for cluster and parallel environment handling."""

    _model_comm_group: torch.distributed.ProcessGroup | None = None  # type: ignore

    def __init__(self, context: Context, **kwargs: Any) -> None:
        """Cluster class for parallel inference

        Parameters
        ----------
        context : Context
            Runner context
        kwargs : Any
            Additional keyword arguments
        """
        self.context = context
        _ = kwargs  # To avoid unused variable warning

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

    def init_process_group(self) -> torch.distributed.ProcessGroup:  # type: ignore
        import torch.distributed as dist

        return dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            timeout=datetime.timedelta(minutes=3),
            world_size=self.world_size,
            rank=self.global_rank,
        )

    def initialise(self) -> None:  # type: ignore
        """Initialise the process group for distributed computing."""
        import torch.distributed as dist

        if self.world_size > 1:
            LOG.info(
                f"Creating a model communication group with {self.world_size} devices with the {self.backend!r} backend"
            )
            model_comm_group = self.init_process_group()

            model_comm_group_ranks = np.arange(self.world_size, dtype=int)
            model_comm_group = dist.new_group(model_comm_group_ranks)
        else:
            model_comm_group = None

        self._model_comm_group = model_comm_group

    @property
    def model_comm_group(self) -> torch.distributed.ProcessGroup | None:  # type: ignore
        """Return the model communication group."""
        return self._model_comm_group

    def seed(self) -> None:
        seed = None
        seed_threshold = 1000
        env_var = "ANEMOI_BASE_SEED"

        if env_var in os.environ:
            seed = int(os.environ[env_var])
            if seed < seed_threshold:
                seed *= seed_threshold  # Ensure seed is sufficiently large

        if self.global_rank == 0:
            seed = seed or torch.initial_seed()
            seed_list = [seed]
            torch.distributed.broadcast_object_list(seed_list, src=0, group=self.model_comm_group)
        else:
            seed_list = [None]
            torch.distributed.broadcast_object_list(seed_list, src=0, group=self.model_comm_group)
            seed = seed_list[0]

        torch.manual_seed(seed)

    def teardown(self) -> None:
        """Tear down the cluster environment."""
        if self.model_comm_group is not None:
            torch.distributed.destroy_process_group()

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
