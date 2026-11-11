# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.inference.lazy import torch
from anemoi.inference.runner import Runner
from anemoi.inference.types import FloatArray

from . import runner_registry

LOG = logging.getLogger(__name__)

# TODO: integrate in base no-model runner


class RicherBatchNoModelMixing:
    @cached_property
    def model(self) -> "torch.nn.Module":

        checkpoint = self.checkpoint  # type: ignore
        multi_metadata = checkpoint.multi_dataset_metadata

        class NoModel(torch.nn.Module):
            """Dummy model class for testing purposes."""

            def __init__(self):
                super().__init__()

            def predict_step(
                self, input_tensors: dict[str, FloatArray] | FloatArray, target_template: dict, **kwargs: Any
            ) -> Any:
                result = {}
                for dataset, data in target_template.items():
                    result[dataset] = data.copy()

                for name, metadata in multi_metadata.items():
                    result[name]["data"] = torch.ones(
                        *metadata.output_shape,
                        dtype=input_tensors[name]["data"].dtype,
                        device=input_tensors[name]["data"].device,
                    )

                return result

        return NoModel()


@runner_registry.register("richer-batch-no-model")
class RicherBatchNoModelRunner(RicherBatchNoModelMixing, Runner):
    pass
