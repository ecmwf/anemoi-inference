# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from functools import cached_property
from typing import Any

from anemoi.inference.lazy import torch
from anemoi.inference.runner import Runner
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from . import runner_registry

LOG = logging.getLogger(__name__)


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


@runner_registry.register("richer-batch")
class RicherBatchRunner(Runner):
    """Injects rich-batch information into the model input."""

    def predict_step(
        self,
        model: "torch.nn.Module",
        input_tensors_torch: dict[str, "torch.Tensor"],
        input_states: dict[str, "State"],
        **kwargs: Any,
    ) -> dict[str, "torch.Tensor"]:
        for key, value in self.config.predict_kwargs.items():
            if key in kwargs:
                warnings.warn(
                    f"`predict_kwargs` contains illegal kwarg `{key}`. This kwarg is set by the runner and will be ignored."
                )
                continue
            kwargs[key] = value

        rich_batch = {}
        target_template = {}

        for dataset, data in input_tensors_torch.items():
            rich_batch[dataset] = {}
            target_template[dataset] = {}

            rich_batch[dataset]["data"] = data
            rich_batch[dataset]["data_type"] = "gridded"
            rich_batch[dataset]["variables"] = list(
                self.tensor_handlers[dataset].metadata.variable_to_input_tensor_index.keys()
            )
            rich_batch[dataset]["latitudes"] = input_states[dataset]["latitudes"]
            rich_batch[dataset]["longitudes"] = input_states[dataset]["longitudes"]
            rich_batch[dataset]["layout"] = ("time", "ensemble", "grid", "variables")

            for key in ("latitudes", "longitudes", "layout", "data_type"):
                target_template[dataset][key] = rich_batch[dataset][key]
            target_template[dataset]["variables"] = list(
                self.tensor_handlers[dataset].metadata.variable_to_output_tensor_index.keys()
            )

        rich_batch_predict = model.predict_step(rich_batch, target_template=target_template, **kwargs)

        result = {}
        for dataset, rich_batch_output in rich_batch_predict.items():
            result[dataset] = rich_batch_output["data"]

        return result


@runner_registry.register("richer-batch-no-model")
class RicherBatchNoModelRunner(RicherBatchNoModelMixing, RicherBatchRunner):
    pass
