# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import os
import socket
import subprocess

import numpy as np
import torch
import torch.distributed as dist

from ..outputs import create_output
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("parallel")
class ParallelRunner(DefaultRunner):
    """Runner which splits a model over multiple devices"""

    def __init__(self, context):
        super().__init__(context)
        global_rank, local_rank, world_size = self.__get_parallel_info()
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

        if self.device == "cuda":
            self.device = f"{self.device}:{local_rank}"
            torch.cuda.set_device(local_rank)

        # disable most logging on non-zero ranks
        if self.global_rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

        # Create a model comm group for parallel inference
        # A dummy comm group is created if only a single device is in use
        model_comm_group = self.__init_parallel(self.device, self.global_rank, self.world_size)
        self.model_comm_group = model_comm_group

        # Ensure each parallel model instance uses the same seed
        if self.global_rank == 0:
            seed = torch.initial_seed()
            torch.distributed.broadcast_object_list([seed], src=0, group=model_comm_group)
        else:
            msg_buffer = np.array([1], dtype=np.uint64)
            torch.distributed.broadcast_object_list(msg_buffer, src=0, group=model_comm_group)
            seed = msg_buffer[0]
            torch.manual_seed(seed)

    def predict_step(self, model, input_tensor_torch, fcstep, **kwargs):
        if self.model_comm_group is None:
            return model.predict_step(input_tensor_torch)
        else:
            try:
                return model.predict_step(input_tensor_torch, self.model_comm_group)
            except TypeError as err:
                LOG.error("Please upgrade to a newer version of anemoi-models to use parallel inference")
                raise err

    def create_output(self):
        if self.global_rank == 0:
            output = create_output(self, self.config.output)
            LOG.info("Output: %s", output)
            return output
        else:
            output = create_output(self, "none")
            return output

    def __del__(self):
        if self.model_comm_group is not None:
            dist.destroy_process_group()

    def __init_network(self):
        """Reads Slurm environment to set master address and port for parallel communication"""

        # Get the master address from the SLURM_NODELIST environment variable
        slurm_nodelist = os.environ.get("SLURM_NODELIST")
        if not slurm_nodelist:
            raise ValueError("SLURM_NODELIST environment variable is not set.")

        # Check if MASTER_ADDR is given, otherwise try set it using 'scontrol'
        master_addr = os.environ.get("MASTER_ADDR")
        if master_addr is None:
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

        # Check if MASTER_PORT is given, otherwise generate one based on SLURM_JOBID
        master_port = os.environ.get("MASTER_PORT")
        if master_port is None:
            LOG.debug("'MASTER_PORT' environment variable not set. Trying to set via SLURM")
            slurm_jobid = os.environ.get("SLURM_JOBID")
            if not slurm_jobid:
                raise ValueError("SLURM_JOBID environment variable is not set.")

            master_port = str(10000 + int(slurm_jobid[-4:]))

        # Print the results for confirmation
        LOG.debug(f"MASTER_ADDR: {master_addr}")
        LOG.debug(f"MASTER_PORT: {master_port}")

        return master_addr, master_port

    def __init_parallel(self, device, global_rank, world_size):
        """Creates a model communication group to be used for parallel inference"""

        if world_size > 1:

            master_addr, master_port = self.__init_network()

            # use 'startswith' instead of '==' in case device is 'cuda:0'
            if device.startswith("cuda"):
                backend = "nccl"
            else:
                backend = "gloo"

            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                timeout=datetime.timedelta(minutes=3),
                world_size=world_size,
                rank=global_rank,
            )
            LOG.info(f"Creating a model comm group with {world_size} devices with the {backend} backend")

            model_comm_group_ranks = np.arange(world_size, dtype=int)
            model_comm_group = torch.distributed.new_group(model_comm_group_ranks)
        else:
            model_comm_group = None

        return model_comm_group

    def __get_parallel_info(self):
        """Reads Slurm env vars, if they exist, to determine if inference is running in parallel"""
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))  # Rank within a node, between 0 and num_gpus
        global_rank = int(os.environ.get("SLURM_PROCID", 0))  # Rank within all nodes
        world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of processes

        return global_rank, local_rank, world_size
