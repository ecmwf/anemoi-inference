# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.lazy import torch
from anemoi.inference.types import FloatArray

from ..runner import RunnerClasses
from ..tensors import TensorHandler
from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class NoModelMixing:
    @cached_property
    def model(self) -> "torch.nn.Module":

        checkpoint: Checkpoint = self.checkpoint  # type: ignore
        multi_metadata = checkpoint.multi_dataset_metadata

        class NoModel(torch.nn.Module):
            """Dummy model class for testing purposes."""

            def __init__(self):
                super().__init__()

            def predict_step(self, input_tensors: dict[str, FloatArray] | FloatArray, **kwargs: Any) -> Any:
                legacy = not isinstance(input_tensors, dict)
                if legacy:
                    input_tensors = {next(iter(multi_metadata.keys())): input_tensors}

                output = {}
                for name, metadata in multi_metadata.items():
                    output[name] = torch.ones(
                        *metadata.output_shape,
                        dtype=input_tensors[name].dtype,
                        device=input_tensors[name].device,
                    )

                if legacy:
                    return next(iter(output.values()))
                return output

        return NoModel()


@runner_registry.register("no-model")
class NoModelRunner(NoModelMixing, DefaultRunner):
    """Runner for running tests.

    Inherits from DefaultRunner.
    """


class SteadyStateTensorHandler(TensorHandler):
    """TensorHandler that skips loading dynamic forcings, keeping them frozen at their initial values."""

    def add_dynamic_forcings_to_input_tensor(self, input_tensor_torch, state, dates, check):
        # Keep existing tensor values (frozen from initial state), just mark slots as set
        for source in self.dynamic_forcings_providers:
            check[source.mask] = True
        return input_tensor_torch


@runner_registry.register("steady-state")
class SteadyStateRunner(DefaultRunner):
    """Runner that uses the real model but freezes dynamic forcings at their initial values.

    Dynamic forcings are not reloaded each step, keeping them at the values
    from the initial state. This replicates the old development_hacks approach.
    """

    def __init__(self, config, *, classes=None) -> None:
        super().__init__(config, classes=RunnerClasses(tensor_handler=SteadyStateTensorHandler))
