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
import warnings
from typing import Any

from anemoi.utils.logs import enable_logging_name

from anemoi.inference.clusters import create_cluster
from anemoi.inference.clusters.client import ComputeClient
from anemoi.inference.clusters.spawner import ComputeSpawner
from anemoi.inference.config import Configuration
from anemoi.inference.lazy import torch
from anemoi.inference.output import Output

from ..decorators import main_argument
from ..outputs import create_output
from ..runner import Runner
from ..runners import create_runner
from . import runner_registry

LOG = logging.getLogger(__name__)


def create_parallel_runner(config: Configuration) -> None:
    """Creates and runs a parallel runner.

    Parameters
    ----------
    config : Configuration
        The configuration object for the runner.
    """
    runner = create_runner(config)
    runner.execute()
    torch.distributed.destroy_process_group()


class NoOp:
    """No operation class used when returning after spawning processes."""

    def execute(self, *a, **k) -> None:
        return None


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

        kwargs = kwargs.copy()
        cluster_config = kwargs.pop("cluster", {})
        compute_client = create_cluster(cluster_config)
        LOG.info(f"Using compute client: {compute_client!r}")

        if isinstance(compute_client, ComputeSpawner):
            with compute_client:
                compute_client.spawn(create_parallel_runner, config)
            return NoOp()

        return ParallelRunner(config, *args, compute_client=compute_client.create_client(), **kwargs)

    @staticmethod
    def get_class(base_class: Runner):
        """Returns a ParallelRunner class object of the given base class."""
        return type("ParallelRunner", (ParallelRunnerMixin, base_class), {})


class ParallelRunnerMixin:
    """Runner which splits a model over multiple devices. Should be mixed in with a base runner class."""

    def __init__(self, config: Any, compute_client: ComputeClient | None = None, **kwargs) -> None:
        """Initialises the ParallelRunner.

        Parameters
        ----------
        config : Any
            The config for the runner.
        compute_client : ComputeClient, optional
            The compute client to use for distributed inference
        """

        super().__init__(config, **kwargs)

        compute_client = compute_client or create_cluster(config.cluster or {}).create_client()  # type: ignore
        assert isinstance(compute_client, ComputeClient), "Compute client must be an instance of ComputeClient."

        LOG.info(f"Using compute client: {compute_client!r}")

        # Set up logging name based on actual cluster rank
        enable_logging_name(f"rank{compute_client.global_rank:02d}")

        self.compute_client = compute_client

        # give the base class an opportunity to modify the parallel runner
        super()._configure_parallel_runner()

        if self.device.type == "cuda":
            self.device = torch.device("cuda", index=compute_client.local_rank)
            torch.cuda.set_device(self.device)
            LOG.info(f"ParallelRunner changing to device `{self.device}`")
        else:
            LOG.info(f"ParallelRunner device `{self.device}` is unchanged")

        self.compute_client = compute_client
        self.is_master = compute_client.is_master
        self.seed(compute_client.process_group)

        # disable most logging on non-zero ranks
        if not self.is_master and self.verbosity == 0:
            LOG.info("ParallelRunner logging disabled on non-zero rank")
            logging.getLogger().setLevel(logging.WARNING)
            warnings.filterwarnings("ignore")

    def seed(self, comm_group: "torch.distributed.ProcessGroup | None") -> None:
        """Seed all processes in the cluster to ensure reproducibility."""
        seed = None
        seed_threshold = 1000
        env_var = "ANEMOI_BASE_SEED"

        if env_var in os.environ:
            seed = int(os.environ[env_var])
            if seed < seed_threshold:
                seed *= seed_threshold  # Ensure seed is sufficiently large

        if self.is_master:
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

        if self.compute_client.process_group is None:
            return super().predict_step(model, input_tensor_torch, **kwargs)
        else:
            try:
                return super().predict_step(
                    model, input_tensor_torch, model_comm_group=self.compute_client.process_group, **kwargs
                )
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
        if self.is_master:
            return super().create_output()
        else:
            output = create_output(self, "none")
            return output
