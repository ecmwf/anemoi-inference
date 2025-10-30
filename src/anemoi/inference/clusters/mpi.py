# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster
from anemoi.inference.lazy import torch

LOG = logging.getLogger(__name__)

MPI_MAPPING = EnvMapping(
    local_rank="OMPI_COMM_WORLD_LOCAL_RANK",
    global_rank="OMPI_COMM_WORLD_RANK",
    world_size="OMPI_COMM_WORLD_SIZE",
    master_addr="MASTER_ADDR",
    master_port="MASTER_PORT",
    init_method="tcp://{master_addr}:{master_port}",
)


@cluster_registry.register("mpi")  # type: ignore
class MPICluster(MappingCluster):  # type: ignore
    """MPI cluster that uses MPI environment variables for distributed setup."""

    def __init__(self, use_mpi_backend: bool = False, **kwargs) -> None:
        """Initialise the MPICluster.

        Parameters
        ----------
        use_mpi_backend : bool, optional
            Use the `mpi` backend in torch, by default False
        """
        super().__init__(mapping=MPI_MAPPING, **kwargs)
        self._use_mpi_backend = use_mpi_backend

    @classmethod
    def used(cls) -> bool:
        return MPI_MAPPING.world_size in os.environ or "PMI_SIZE" in os.environ

    @property
    def backend(self) -> str:
        """Return the backend string for distributed computing."""
        if self._use_mpi_backend:
            return "mpi"
        return super().backend

    def create_model_comm_group(self) -> torch.distributed.ProcessGroup | None:
        """Create the communication group for model parallelism."""
        if not self._use_mpi_backend:
            return super().create_model_comm_group()

        if self.world_size <= 1:
            return None

        import torch.distributed as dist

        LOG.info("Creating model communication group for parallel inference")
        group = dist.init_process_group(
            backend=self.backend,
        )

        # Create a new process group for model communication
        group = dist.new_group(
            ranks=list(range(self.world_size)),
        )
        LOG.info("Model communication group created")

        return group
