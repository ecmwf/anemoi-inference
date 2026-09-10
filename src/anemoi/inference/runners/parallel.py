# (C) Copyright 2025-2026 Anemoi contributors.
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
from anemoi.inference.clusters.client import ComputeClientFactory
from anemoi.inference.clusters.spawner import ComputeSpawner
from anemoi.inference.config import Configuration
from anemoi.inference.lazy import torch
from anemoi.inference.output import Output

from ..decorators import main_argument
from ..outputs import create_output
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


def _gather_grid_tensor_to_master(
    tensor: "torch.Tensor",
    shard_sizes: list[int],
    process_group: "torch.distributed.ProcessGroup",
    is_master: bool,
) -> "torch.Tensor | None":
    """Gather uneven grid shards on rank 0 without materialising the full tensor elsewhere."""
    max_shard_size = max(shard_sizes)
    padded_shape = list(tensor.shape)
    padded_shape[0] = max_shard_size
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.shape[0]].copy_(tensor)

    gathered = [torch.empty_like(padded) for _ in shard_sizes] if is_master else None
    torch.distributed.gather(padded, gather_list=gathered, dst=0, group=process_group)

    if not is_master:
        return None

    return torch.cat([shard[:size] for shard, size in zip(gathered, shard_sizes)], dim=0)


def create_parallel_runner(config: Configuration, client_factory: ComputeClientFactory) -> None:
    """Creates and runs a parallel runner.

    Parameters
    ----------
    config : Configuration
        The configuration object for the runner.
    client_factory : ComputeClientFactory
        The compute client factory to use for distributed inference.
    """
    runner_config: dict[str, Any] = config.runner.get("parallel", {})  # type: ignore
    if isinstance(runner_config, str):
        runner_config = {"base_runner": runner_config}
    runner_config["cluster"] = client_factory.create_client()

    runner = ParallelRunnerFactory(config, **runner_config)  # type: ignore
    runner.execute()


class NoOp:
    """No operation class used when returning after spawning processes."""

    def execute(self, *a, **k) -> None:
        return None


@runner_registry.register("parallel")
@main_argument("base_runner")
class ParallelRunnerFactory:
    """Creates a ParallelRunner with a dynamic base class.

    Parameters
    ----------
    config : Any
        The config for the runner.
    base_runner : str
        The base runner to use for the parallel runner.
        Must subclass from at least `DefaultRunner`.
    cluster : str | dict[str, str] | ComputeClient | None, optional
        The cluster configuration or instance to use for distributed inference, by default None
    """

    def __new__(
        cls,
        config: Any,
        base_runner: str = "default",
        *args,
        cluster: str | dict[str, str] | ComputeClient | None = None,
        **kwargs,
    ):
        assert base_runner != "parallel", "Base runner cannot be `parallel` itself."

        try:
            base_class = runner_registry.lookup(base_runner)
        except ValueError:
            raise ValueError(f"Base runner '{base_runner}' not found in the registry.")

        assert issubclass(base_class, Runner), f"Base runner '{base_runner}' must be a subclass of Runner."

        LOG.debug(f"Creating ParallelRunner from base runner: {base_runner} ({base_class.__name__})")

        ParallelRunner = cls.get_class(base_class)
        if not isinstance(cluster, (ComputeClient,)):
            compute = create_cluster(cluster or {})
        else:
            compute = cluster

        if isinstance(compute, ComputeSpawner):
            with compute:
                compute.spawn(create_parallel_runner, config)
            return NoOp()

        compute_client = compute if isinstance(compute, ComputeClient) else compute.create_client()

        LOG.info(f"Using compute client provider: {compute!r}")
        return ParallelRunner(config, *args, compute_client=compute_client, **kwargs)

    @staticmethod
    def get_class(base_class: Runner):
        """Returns a ParallelRunner class object of the given base class."""
        return type("ParallelRunner", (ParallelRunnerMixin, base_class), {})


class ParallelRunnerMixin(Runner):
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

        compute_client = compute_client or create_cluster(config.cluster or {}).create_client()  # type: ignore
        assert isinstance(compute_client, ComputeClient), "Compute client must be an instance of ComputeClient."

        # Set up logging name based on actual cluster rank
        enable_logging_name(f"rank{compute_client.global_rank:02d}")
        LOG.info(f"{compute_client!r}")

        self.compute_client = compute_client
        self.is_master = compute_client.is_master
        self.grid_shard_sizes: dict[str, list[int]] = {}

        super().__init__(config, **kwargs)

        # give the base class an opportunity to modify the parallel runner
        super()._configure_parallel_runner()

        if self.device.type == "cuda":
            self.device = torch.device("cuda", index=compute_client.local_rank)
            torch.cuda.set_device(self.device)
            LOG.debug(f"ParallelRunner changing to device `{self.device}`")
        else:
            LOG.warning(f"ParallelRunner device `{self.device}` is unchanged")

        self.seed(compute_client.process_group)

        # disable most logging on non-zero ranks
        if not self.is_master and self.verbosity == 0:
            LOG.debug("ParallelRunner logging disabled on non-zero rank")
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

    def prepare_forecast_input_tensors(
        self, input_tensors_torch: dict[str, "torch.Tensor"]
    ) -> dict[str, "torch.Tensor"]:
        """Shard full-grid input tensors once before autoregressive inference."""
        process_group = self.compute_client.process_group
        if process_group is None:
            return input_tensors_torch

        from anemoi.models.distributed.graph import shard_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes

        for dataset, tensor in input_tensors_torch.items():
            shard_sizes = get_shard_sizes(tensor, -2, model_comm_group=process_group)
            assert shard_sizes is not None
            self.grid_shard_sizes[dataset] = shard_sizes
            input_tensors_torch[dataset] = shard_tensor(tensor, -2, shard_sizes, process_group)

        return input_tensors_torch

    def grid_shard_slice(self, dataset: str) -> slice:
        """Return this rank's interval in the full grid."""
        shard_sizes = self.grid_shard_sizes.get(dataset)
        if shard_sizes is None:
            return slice(None)

        process_group = self.compute_client.process_group
        assert process_group is not None
        rank = process_group.rank()
        start = sum(shard_sizes[:rank])
        return slice(start, start + shard_sizes[rank])

    def prepare_forecast_output_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Gather all forecast fields together on rank 0 while keeping rollout state sharded."""
        process_group = self.compute_client.process_group
        if process_group is None:
            return state

        result = {}
        for dataset, dataset_state in state.items():
            output_state = dataset_state.copy()
            output_state["fields"] = {}
            shard_sizes = self.grid_shard_sizes[dataset]
            fields = dataset_state["fields"]
            if not fields:
                result[dataset] = output_state
                continue

            field_names = list(fields)
            field_tensors = list(fields.values())
            grid_sizes = {field.shape[0] for field in field_tensors}
            full_grid_size = sum(shard_sizes)
            rank = process_group.rank()

            if grid_sizes == {full_grid_size}:
                if self.is_master:
                    output_state["fields"] = fields.copy()
                result[dataset] = output_state
                continue

            if grid_sizes != {shard_sizes[rank]}:
                raise ValueError(
                    f"[{dataset}] fields have grid sizes {sorted(grid_sizes)}, expected "
                    f"local shard size {shard_sizes[rank]} or full grid size {full_grid_size}"
                )

            if any(field.ndim != 1 for field in field_tensors):
                shapes = {name: tuple(field.shape) for name, field in fields.items()}
                raise ValueError(f"[{dataset}] fields must be one-dimensional grid tensors, got {shapes}")

            dtypes = {field.dtype for field in field_tensors}
            devices = {field.device for field in field_tensors}
            if len(dtypes) != 1 or len(devices) != 1:
                raise ValueError(
                    f"[{dataset}] fields must share one dtype and device for a packed gather, "
                    f"got dtypes={dtypes}, devices={devices}"
                )

            packed_fields = torch.stack(field_tensors, dim=1)
            LOG.debug(
                "Rank %d gathering %d [%s] fields with packed local shape %s",
                rank,
                len(field_names),
                dataset,
                tuple(packed_fields.shape),
            )
            gathered = _gather_grid_tensor_to_master(packed_fields, shard_sizes, process_group, self.is_master)
            if self.is_master:
                output_state["fields"] = {name: gathered[:, index] for index, name in enumerate(field_names)}
            result[dataset] = output_state

        return result

    def write_output_state(self, dataset: str, state: dict[str, Any]) -> None:
        """Only rank 0 owns full-grid forecast output."""
        if self.is_master:
            super().write_output_state(dataset, state)

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
                if self.grid_shard_sizes:
                    kwargs["grid_shard_sizes"] = self.grid_shard_sizes
                    kwargs["gather_out"] = False
                return super().predict_step(
                    model,
                    input_tensor_torch,
                    model_comm_group=self.compute_client.process_group,
                    **kwargs,
                )
            except TypeError as err:
                LOG.error(
                    "Please upgrade to a newer version of anemoi-models (at least version v0.4.2) to use parallel inference. If updating breaks your checkpoints, you can try reverting to your original version of anemoi-models and cherry-picking 'https://github.com/ecmwf/anemoi-core/pull/77'"
                )
                raise err

    def complete_forecast_hook(self) -> None:
        """Hook called at the end of the forecast."""
        super().complete_forecast_hook()
        torch.distributed.destroy_process_group()

    def create_output(self, *args, **kwargs) -> Output:
        """Creates the real output on rank 0 and a `none` on the others."""
        if self.is_master:
            return super().create_output(*args, **kwargs)
        else:
            # passing the metadata here is a bit of a hack, the `none` output doesn't really need it
            # but in multi-datasets world every output needs metadata, so do this workaround
            output = create_output(self, "none", self.checkpoint._metadata)
            return output
