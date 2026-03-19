# (C) Copyright 2025 Anemoi contributors.
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

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class NoModelMixing:
    @cached_property
    def model(self) -> "torch.nn.Module":

        checkpoint: Checkpoint = self.checkpoint
        multi_metadata = checkpoint.multi_dataset_metadata

        class NoModel(torch.nn.Module):
            """Dummy model class for testing purposes."""

            def __init__(self):
                super().__init__()

            def predict_step(self, input_tensors: dict[str, FloatArray], **kwargs: Any) -> Any:
                output = {}
                for name, metadata in multi_metadata.items():
                    input_shape = input_tensors[name].shape
                    input_tensor = input_tensors[name]
                    output_shape = (
                        input_shape[0],  # batch
                        metadata.multi_step_output,  # time
                        1,  # ensemble
                        input_shape[2],  # gridpoints
                        len(metadata.output_tensor_index_to_variable),  # variables
                    )

                    output[name] = torch.ones(*output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

                return output

        return NoModel()


@runner_registry.register("no-model")
class NoModelRunner(NoModelMixing, DefaultRunner):
    """Runner for running tests.

    Inherits from DefaultRunner.
    """
