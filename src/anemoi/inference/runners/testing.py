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

import rich

from anemoi.inference.lazy import torch

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class TestingMixing:
    # Used with dummy checkpoint (see pytests)
    def predict_step(self, model: Any, input_tensor_torch: Any, **kwargs: Any) -> Any:
        """Perform a prediction step using the model.

        Parameters
        ----------
        model : Any
            The model to use for prediction.
        input_tensor_torch : torch.Tensor
            The input tensor for the model.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The prediction result.
        """
        return model.predict_step(input_tensor_torch, **kwargs)


class NoModelMixing:
    # Use with a real checkpoint (see prepml/regression)
    def predict_step(self, model: Any, input_tensor_torch: Any, **kwargs: Any) -> Any:
        """Perform a prediction step using the model.

        Parameters
        ----------
        model : Any
            The model to use for prediction.
        input_tensor_torch : torch.Tensor
            The input tensor for the model.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The prediction result.
        """
        return model.predict_step(input_tensor_torch, **kwargs)

    @cached_property
    def model(self) -> "torch.nn.Module":

        checkpoint = self.checkpoint
        number_of_output_variables = len(checkpoint.output_tensor_index_to_variable)
        rich.print(f"Loading model from checkpoint: {checkpoint.output_tensor_index_to_variable}")

        class NoModel(torch.nn.Module):
            """Dummy model class for testing purposes."""

            def __init__(self):
                super().__init__()

            def predict_step(self, input_tensor: Any, **kwargs: Any) -> Any:
                input_shape = input_tensor.shape
                output_shape = (
                    input_shape[0],  # batch
                    1,  # time
                    input_shape[2],  # gridpoints
                    number_of_output_variables,  # variables
                )

                # return torch.rand(*output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
                return torch.zeros(*output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

        return NoModel()


@runner_registry.register("testing")
class TestingRunner(TestingMixing, DefaultRunner):
    """Runner for running tests.

    Inherits from DefaultRunner.
    """


@runner_registry.register("no-model")
class NoModelRunner(NoModelMixing, DefaultRunner):
    """Runner for running tests.

    Inherits from DefaultRunner.
    """
