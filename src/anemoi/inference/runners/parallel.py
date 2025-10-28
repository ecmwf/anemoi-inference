# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

from anemoi.utils.logs import enable_logging_name

from anemoi.inference.clusters import Cluster
from anemoi.inference.clusters import create_cluster
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


@runner_registry.register("parallel")  # type: ignore
@main_argument("base_runner")
class ParallelRunnerFactory:
    """Creates a ParallelRunner with a dynamic base class."""

    def __new__(cls, config: Any, base_runner: str = "default", *args, **kwargs):
        assert base_runner != "parallel", "Base runner cannot be `parallel` itself."

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


class ParallelRunnerMixin(Runner):
    """Runner which splits a model over multiple devices. Should be mixed in with a base runner class."""

    def __new__(cls, config, *args, **kwargs):

        if torch.cuda.is_available():
            return super().__new__(cls)
        else:
            LOG.warning("CUDA is not available. Falling back to DefaultRunner")
            return DefaultRunner(config)

    def __init__(self, config: Any, pid: int = 0, cluster: Cluster | None = None, **kwargs) -> None:
        """Initialises the ParallelRunner.

        Parameters
        ----------
        config : Any
            The config for the runner.
        cluster : Cluster, optional
            The cluster to use for distributed inference. If None, a cluster is created based on the config.
        pid : int, optional
            The process ID, by default 0.
        """

        super().__init__(config, **kwargs)

        self.model_comm_group = None

        # give the base class an opportunity to modify the parallel runner
        super()._configure_parallel_runner()

        self.cluster = cluster or create_cluster(self, config.cluster or {})
        self.cluster.configure(pid=pid)

        LOG.info(f"Using cluster: {self.cluster!r}")

        # Set up logging name based on actual cluster rank
        enable_logging_name(f"rank{self.cluster.global_rank:02d}")

        self.cluster.spawn(create_parallel_runner, config)  # TODO: How to split concern?
        self.model_comm_group = self.create_model_comm_group(self.cluster)

        if self.device.type == "cuda":
            self.device = torch.device("cuda", index=self.cluster.local_rank)
            torch.cuda.set_device(self.device)
            LOG.info(f"ParallelRunner changing to device `{self.device}`")
        else:
            LOG.info(f"ParallelRunner device `{self.device}` is unchanged")

        self.seed(self.model_comm_group)

        # disable most logging on non-zero ranks
        if not self.cluster.is_master and self.verbosity == 0:
            LOG.info("ParallelRunner logging disabled on non-zero rank")
            logging.getLogger().setLevel(logging.WARNING)

    def create_model_comm_group(self, cluster: Cluster) -> torch.distributed.ProcessGroup | None:
        """Create a model communication group for the cluster.

        Parameters
        ----------
        cluster : Cluster
            The cluster to use for the communication group.

        Returns
        -------
        torch.distributed.ProcessGroup | None
            The created communication group, or None if not applicable.
        """
        comm_group_init = cluster.comm_group_init
        if comm_group_init.world_size <= 1:
            return None

        import torch.distributed as dist

        LOG.info("Creating model communication group for parallel inference")
        group = dist.init_process_group(**comm_group_init.init_kwargs)

        # Create a new process group for model communication
        group = dist.new_group(
            ranks=list(range(comm_group_init.world_size)),
        )
        LOG.info("Model communication group created")

        return group

    def seed(self, comm_group: "torch.distributed.ProcessGroup | None") -> None:
        """Seed all processes in the cluster to ensure reproducibility."""
        seed = None
        seed_threshold = 1000
        env_var = "ANEMOI_BASE_SEED"

        if env_var in os.environ:
            seed = int(os.environ[env_var])
            if seed < seed_threshold:
                seed *= seed_threshold  # Ensure seed is sufficiently large

        if self.cluster.global_rank == 0:
            seed = seed or torch.initial_seed()
            seed_list = [seed]
            torch.distributed.broadcast_object_list(seed_list, src=0, group=comm_group)
        else:
            seed_list = [None]
            torch.distributed.broadcast_object_list(seed_list, src=0, group=comm_group)
            seed = seed_list[0]

        torch.manual_seed(seed)

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
        if self.cluster.is_master:
            return super().create_output()
        else:
            output = create_output(self, "none")
            return output

    def __del__(self):
        """Cleans up the model communication group on deletion."""
        if self.model_comm_group is not None:
            torch.distributed.destroy_process_group()
