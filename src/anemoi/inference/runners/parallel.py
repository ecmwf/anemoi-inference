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
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import numpy as np
from anemoi.utils.logs import enable_logging_name

from anemoi.inference.config import Configuration
from anemoi.inference.lazy import torch
from anemoi.inference.output import Output

from ..decorators import main_argument
from ..outputs import create_output
from ..runner import Runner
from ..runners import create_runner
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


def create_parallel_runner(config: Configuration, pid: int) -> None:
    """Creates and runs a parallel runner.

    Parameters
    ----------
    config : Configuration
        The configuration object for the runner.
    pid : int
        The process ID.
    """
    runner = create_runner(config, pid=pid)
    runner.execute()


@runner_registry.register("parallel")
@main_argument("base_runner")
class ParallelRunnerFactory:
    """Creates a ParallelRunner with a dynamic base class."""

    def __new__(cls, config: Any, base_runner: str = "default", *args, **kwargs):
        assert base_runner != "parallel", "Base runner cannot be `parallel` itself."

        enable_logging_name(f"rank{int(os.environ.get('SLURM_PROCID',0)):02d}")

        try:
            base_class = runner_registry.lookup(base_runner)
        except ValueError:
            raise ValueError(f"Base runner '{base_runner}' not found in the registry.")

        assert issubclass(base_class, Runner), f"Base runner '{base_runner}' must be a subclass of Runner."

        LOG.info(f"Creating ParallelRunner from base runner: {base_runner} ({base_class.__name__})")

        ParallelRunner = cls.get_class(base_class)
        return ParallelRunner(config, *args, **kwargs)

    @staticmethod
    def get_class(base_class: Runner):
        """Returns a ParallelRunner class object of the given base class."""
        return type("ParallelRunner", (ParallelRunnerMixin, base_class), {})


class ParallelRunnerMixin:
    """Runner which splits a model over multiple devices. Should be mixed in with a base runner class."""

    def __new__(cls, config, *args, **kwargs):

        if torch.cuda.is_available():
            return super().__new__(cls)
        else:
            LOG.warning("CUDA is not available. Falling back to DefaultRunner")
            return DefaultRunner(config)

    def __init__(self, config: Any, pid: int = 0, **kwargs) -> None:
        """Initializes the ParallelRunner.

        Parameters
        ----------
        config : Any
            The config for the runner.
        pid : int, optional
            The process ID, by default 0.
        """

        super().__init__(config, **kwargs)

        self.model_comm_group = None
        self.pid = pid

        # give the base class an opportunity to modify the parallel runner
        super()._configure_parallel_runner()

        self._bootstrap_processes()

        LOG.info(
            f"ParallelRunner local/global ranks: {self.local_rank}/{self.global_rank}, host: {socket.gethostname()}"
        )

        if self.device.type == "cuda":
            self.device = torch.device("cuda", index=self.local_rank)
            torch.cuda.set_device(self.device)
            LOG.info(f"ParallelRunner changing to device `{self.device}`")
        else:
            LOG.info(f"ParallelRunner device `{self.device}` is unchanged")

        # disable most logging on non-zero ranks
        if self.global_rank != 0 and self.verbosity == 0:
            LOG.info("ParallelRunner logging disabled on non-zero rank")
            logging.getLogger().setLevel(logging.WARNING)

        # Create a model comm group for parallel inference
        # A dummy comm group is created if only a single device is in use
        if self.world_size > 1:
            model_comm_group = self._init_parallel()
            self.model_comm_group = model_comm_group

            # Ensure each parallel model instance uses the same seed
            self._seed_procs()
        else:
            LOG.warning("ParallelRunner selected but world size of 1 detected")

    def predict_step(self, model: Any, input_tensor_torch: "torch.Tensor", **kwargs: Any) -> "torch.Tensor":
        """Performs a prediction step.

        Parameters
        ----------
        model : Any
            The model to use for prediction.
        input_tensor_torch : torch.Tensor
            The input tensor for the model.
        **kwargs : Any
            Additional arguments for the prediction step.

        Returns
        -------
        torch.Tensor
            The prediction result.
        """
        # call the predict_step of the base class since it might do some modifications
        # the base class is expected to forward the kwargs (including the comm group) to the model's predict_step method

        if self.model_comm_group is None:
            return super().predict_step(model, input_tensor_torch, **kwargs)
        else:
            try:
                return super().predict_step(model, input_tensor_torch, model_comm_group=self.model_comm_group, **kwargs)
            except TypeError as err:
                LOG.error(
                    "Please upgrade to a newer version of anemoi-models (at least version v0.4.2) to use parallel inference. If updating breaks your checkpoints, you can try reverting to your original version of anemoi-models and cherry-picking 'https://github.com/ecmwf/anemoi-core/pull/77'"
                )
                raise err

    def create_output(self) -> Output:
        """Creates the real output on rank 0 and a `none` on the others.

        Returns
        -------
        Output
            The created output.
        """
        if self.global_rank == 0:
            return super().create_output()
        else:
            output = create_output(self, "none")
            return output

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        if self.model_comm_group is not None:
            torch.distributed.destroy_process_group()

    def _seed_procs(self) -> None:
        """Ensures each process uses the same seed.
        Will try read 'ANEMOI_BASE_SEED' from the environment.
        Otherwise, the seed of process 0 will be shared to all processes.
        """

        seed = None
        seed_threshold = 1000
        env_var_list = ["ANEMOI_BASE_SEED"]
        for env_var in env_var_list:
            if env_var in os.environ:
                seed = int(os.environ.get(env_var))
                if seed < seed_threshold:
                    seed *= seed_threshold  # make it (hopefully) big enough
                break

        if self.global_rank == 0:
            if seed is None:
                seed = torch.initial_seed()
            torch.distributed.broadcast_object_list([seed], src=0, group=self.model_comm_group)
        else:
            msg_buffer = np.array([1], dtype=np.uint64)
            torch.distributed.broadcast_object_list(msg_buffer, src=0, group=self.model_comm_group)
            seed = msg_buffer[0]
            torch.manual_seed(seed)

    def _srun_used(self) -> bool:
        """Checks if anemoi-inference was launched with srun.

        Returns
        -------
        bool
            True if srun is used, False otherwise.
        """
        # from pytorch lightning
        # https://github.com/Lightning-AI/pytorch-lightning/blob/a944e7744e57a5a2c13f3c73b9735edf2f71e329/src/lightning/fabric/plugins/environments/slurm.py
        return "SLURM_NTASKS" in os.environ and os.environ.get("SLURM_JOB_NAME") not in ("bash", "interactive")

    def _spawn_parallel_procs(self, num_procs: int) -> None:
        """When srun is not available, this method creates N-1 child processes within the same node for parallel inference.

        Parameters
        ----------
        num_procs : int
            The number of processes to spawn.
        """
        LOG.debug(f"spawning {num_procs -1 } procs")

        # check num_procs <= num_gpus
        if str(self.device).startswith("cuda"):
            num_gpus = torch.cuda.device_count()
            if num_procs > num_gpus:
                raise ValueError(
                    f"You requested parallel inference over {num_procs} GPUs but your node only has {num_gpus} GPUs available."
                )

        # Create N-1 procs, each with a unique PID
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        config = self.config
        for pid in range(1, num_procs):
            mp.Process(target=create_parallel_runner, args=(config, pid)).start()

    def _bootstrap_processes(self) -> None:
        """Initialises processes and their network information.

        If srun is available, Slurm variables are read to determine network settings.
        Otherwise, local processes are spawned and network info is inferred from the configuration.
        """
        using_slurm = self._srun_used()
        if using_slurm:

            # Determine world size and rank from slurm env vars
            global_rank, local_rank, world_size = self._get_parallel_info_from_slurm()
            self.global_rank = global_rank
            self.local_rank = local_rank
            self.world_size = world_size

            # determine master address and port from slurm/override env vars
            slurm_addr, slurm_port = self._init_network_from_slurm()
            self.master_addr = slurm_addr
            self.master_port = slurm_port

            # the world size entry in the config is only needed when not launching via srun
            if self.config.world_size != 1:
                LOG.warning(
                    f"world size ({self.config.world_size}) set in the config is ignored because we are launching via srun, using 'SLURM_NTASKS' instead"
                )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # New branch for Azure ML / general distributed env (e.g., env:// mode)
            self.global_rank = int(os.environ["RANK"])
            self.local_rank = int(
                os.environ.get("LOCAL_RANK", self.global_rank)
            )  # Fallback to global if LOCAL_RANK unset
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.master_addr = os.environ.get("MASTER_ADDR")
            self.master_port = os.environ.get("MASTER_PORT")
            if self.master_addr is None or self.master_port is None:
                raise ValueError(
                    "MASTER_ADDR and MASTER_PORT must be set for distributed initialization (e.g., in Azure ML)"
                )
            if self.config.world_size != 1 and self.config.world_size != self.world_size:
                LOG.warning(
                    f"Config world_size ({self.config.world_size}) ignored; using WORLD_SIZE from environment ({self.world_size})"
                )
        else:
            # If srun is not available, spawn procs manually on a node

            # Read the config to determine world_size and pid
            self.global_rank = self.pid  # only inference within a single node is supported when not using srun
            self.local_rank = self.pid
            self.world_size = self.config.world_size
            if self.world_size == 1:
                LOG.warning(
                    "You selected 'runner: parallel' but you have only set 'world_size: 1'. Please update world_size or launch via srun to make use of parallel inference"
                )
            if self.world_size <= 0:
                raise ValueError(
                    f"Error. 'world_size' must be greater then 1 to use parallel inference. {self.config.world_size=} set in the config is invalid."
                )

            # since we are running within a node, 'localhost' and any port can be used
            self.master_addr = "localhost"
            # generates a port between 10000 and 19999, based on the nodes hostname (which will be the same across all node-local procs)
            import hashlib

            node_name = os.uname().nodename.encode()  # Convert to bytes
            hash_val = int(hashlib.md5(node_name).hexdigest(), 16)  # Convert hash to int
            self.master_port = 10000 + (hash_val % 9999)

            # Spawn the other processes manually
            if self.local_rank == 0:
                self._spawn_parallel_procs(self.world_size)

    def _init_network_from_slurm(self) -> tuple[str, str]:
        """Reads Slurm environment to set master address and port for parallel communication.

        Returns
        -------
        Tuple[str, str]
            The master address and port.
        """
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

    def _init_parallel(self) -> Optional["torch.distributed.ProcessGroup"]:
        """Creates a model communication group to be used for parallel inference.

        Returns
        -------
        Optional[dist.ProcessGroup]
            The model communication group.
        """
        import torch.distributed as dist

        if self.world_size > 1:

            # use 'startswith' instead of '==' in case device is 'cuda:0'
            if str(self.device).startswith("cuda"):
                backend = "nccl"
            else:
                if dist.is_mpi_available():
                    backend = "mpi"
                else:
                    backend = "gloo"

            if backend == "mpi":
                # MPI backend: No init_method or explicit sizes needed
                dist.init_process_group(backend="mpi")
                model_comm_group_ranks = np.arange(self.world_size, dtype=int)
                model_comm_group = dist.new_group(model_comm_group_ranks)
            else:
                if self._using_distributed_env():
                    init_method = "env://"  # Azure ML recommended
                else:
                    init_method = f"tcp://{self.master_addr}:{self.master_port}"
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    timeout=datetime.timedelta(minutes=3),
                    world_size=self.world_size,
                    rank=self.global_rank,
                )
                model_comm_group_ranks = np.arange(self.world_size, dtype=int)
                model_comm_group = dist.new_group(model_comm_group_ranks)
            LOG.info(f"Creating a model communication group with {self.world_size} devices with the {backend} backend")
        else:
            model_comm_group = None
            LOG.warning("ParallelRunner selected but world size of 1 detected")

        return model_comm_group

    def _get_parallel_info_from_slurm(self) -> tuple[int, int, int]:
        """Reads Slurm env vars, if they exist, to determine if inference is running in parallel.

        Returns
        -------
        Tuple[int, int, int]
            The global rank, local rank, and world size.
        """
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))  # Rank within a node, between 0 and num_gpus
        global_rank = int(os.environ.get("SLURM_PROCID", 0))  # Rank within all nodes
        world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of processes

        return global_rank, local_rank, world_size

    def _using_distributed_env(self) -> bool:
        """Checks for distributed env vars like those in Azure ML."""
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ

    def _is_mpi_env(self) -> bool:
        """Detects common MPI implementations (optional, for generality)."""
        return "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
