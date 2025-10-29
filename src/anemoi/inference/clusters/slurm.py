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
import socket
import subprocess
from functools import cached_property

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster

LOG = logging.getLogger(__name__)

SLURM_MAPPING = EnvMapping(
    local_rank="SLURM_LOCALID",
    global_rank="SLURM_PROCID",
    world_size="SLURM_NTASKS",
    master_addr="MASTER_ADDR",
    master_port="MASTER_PORT",
    init_method="tcp://{master_addr}:{master_port}",
)


@cluster_registry.register("slurm")  # type: ignore
class SlurmCluster(MappingCluster):  # type: ignore
    """Slurm cluster that uses SLURM environment variables for distributed setup."""

    def __init__(self) -> None:
        super().__init__(mapping=SLURM_MAPPING)

    @classmethod
    def used(cls) -> bool:
        # from pytorch lightning
        # https://github.com/Lightning-AI/pytorch-lightning/blob/a944e7744e57a5a2c13f3c73b9735edf2f71e329/src/lightning/fabric/plugins/environments/slurm.py
        return SLURM_MAPPING.world_size in os.environ and os.environ.get("SLURM_JOB_NAME") not in (
            "bash",
            "interactive",
        )

    @property
    def master_addr(self) -> str:
        """Return the master address."""
        # Get the master address from the SLURM_NODELIST environment variable

        slurm_nodelist = os.environ.get("SLURM_NODELIST")
        if not slurm_nodelist:
            raise ValueError("SLURM_NODELIST environment variable is not set.")

        # Check if MASTER_ADDR is given, otherwise try set it using 'scontrol'
        master_addr = super().master_addr
        if not master_addr:
            LOG.debug("'MASTER_ADDR' environment variable not set. Trying to set via SLURM")
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostname", slurm_nodelist], stdout=subprocess.PIPE, text=True, check=True
                )
            except subprocess.CalledProcessError as err:
                LOG.error(
                    "Python could not execute 'scontrol show hostname $SLURM_NODELIST' while calculating MASTER_ADDR. You could avoid this error by setting the MASTER_ADDR env var manually."
                )
                raise err

            master_addr = result.stdout.splitlines()[0]

            # Resolve the master address using nslookup
            try:
                master_addr = socket.gethostbyname(master_addr)
            except socket.gaierror:
                raise ValueError(f"Could not resolve hostname: {master_addr}")

        return master_addr

    @cached_property
    def master_port(self) -> int:
        """Return the master port."""
        # Check if MASTER_PORT is given, otherwise generate one based on SLURM_JOBID

        master_port = super().master_port
        if master_port is None or master_port == 0:
            LOG.debug("'MASTER_PORT' environment variable not set. Trying to set via SLURM")
            slurm_jobid = os.environ.get("SLURM_JOBID")
            if not slurm_jobid:
                raise ValueError("SLURM_JOBID environment variable is not set.")

            master_port = 10000 + int(slurm_jobid[-4:])

        return master_port
