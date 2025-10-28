# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
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

        self.cluster = cluster or create_cluster(self, config.cluster or {}, pid=pid)

        # Set up logging name based on actual cluster rank
        enable_logging_name(f"rank{self.cluster.global_rank:02d}")

        LOG.info(f"Using cluster: {self.cluster!r}")

        self.cluster.spawn(create_parallel_runner, config)
        self.cluster.initialise()

        if self.device.type == "cuda":
            self.device = torch.device("cuda", index=self.cluster.local_rank)
            torch.cuda.set_device(self.device)
            LOG.info(f"ParallelRunner changing to device `{self.device}`")
        else:
            LOG.info(f"ParallelRunner device `{self.device}` is unchanged")

        self.cluster.seed()

        # disable most logging on non-zero ranks
        if not self.cluster.is_master and self.verbosity == 0:
            LOG.info("ParallelRunner logging disabled on non-zero rank")
            logging.getLogger().setLevel(logging.WARNING)

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

        if self.cluster.model_comm_group is None:
            return super().predict_step(model, input_tensor_torch, **kwargs)
        else:
            try:
                return super().predict_step(
                    model, input_tensor_torch, model_comm_group=self.cluster.model_comm_group, **kwargs
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
        if self.cluster.is_master:
            return super().create_output()
        else:
            output = create_output(self, "none")
            return output
