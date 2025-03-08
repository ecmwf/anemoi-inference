# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any
from typing import Dict

import torch
from anemoi.utils.config import DotDict

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.config import LOG
from anemoi.inference.metadata import Metadata
from anemoi.inference.testing import float_hash


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def __init__(self, medatada: Dict[str, Any], supporting_arrays: Dict[str, Any]) -> None:
        """Initialize the mock model.

        Parameters
        ----------
        medatada : dict
            The metadata for the model.
        supporting_arrays : dict
            The supporting arrays for the model.
        """
        super().__init__()
        metadata = DotDict(medatada)

        self.features_in = len(metadata.data_indices.model.input.full)
        self.features_out = len(metadata.data_indices.model.output.full)
        self.roll_window = metadata.config.training.multistep_input
        self.grid_size = metadata.dataset.shape[-1]

        self.input_shape = (1, self.roll_window, self.grid_size, self.features_in)
        self.output_shape = (1, 1, self.grid_size, self.features_out)

        checkpoint = Checkpoint(Metadata(metadata))
        self.input_variables = {v: k for k, v in checkpoint.variable_to_input_tensor_index.items()}
        self.output_variables = checkpoint.output_tensor_index_to_variable
        self.lagged = checkpoint.lagged

        self.typed_variables = checkpoint.typed_variables

        self.first = True

    def predict_step(
        self, x: torch.Tensor, date: datetime.datetime, step: datetime.timedelta, **kwargs: Any
    ) -> torch.Tensor:
        """Perform a prediction step."""
        assert x.shape == self.input_shape, f"Expected {self.input_shape}, got {x.shape}"

        # Date of the data
        tensor_dates = [(date - step) + h for h in self.lagged]

        for lag in range(x.shape[1]):
            for feature in range(x.shape[3]):
                value = float_hash(self.input_variables[feature], tensor_dates[lag])
                expect = torch.Tensor([value])
                variable = self.typed_variables[self.input_variables[feature]]

                if not variable.is_computed_forcing and not variable.is_constant_in_time:
                    # Normal prognostice variable
                    if not torch.isclose(x[:, lag, :, feature], expect).all():
                        LOG.error(
                            "Feature: %s (%s), lag: %s (%s)",
                            feature,
                            self.input_variables[feature],
                            lag,
                            tensor_dates[lag],
                        )
                        LOG.error("Date: %s, Step: %s, Tensor dates: %s", date, step, tensor_dates)
                        LOG.error(
                            "Expected %s for %s at lag %s, got %s",
                            value,
                            self.input_variables[feature],
                            tensor_dates[lag],
                            x[:, lag, :, feature][..., 0],
                        )
                        LOG.error("x: %s", x[:, :, :, feature])

                        raise ValueError(
                            f"Expected {expect} for {self.input_variables[feature]} at lag {lag}, got {x[:, lag, :, feature][...,0]}"
                        )

        y = torch.zeros(self.output_shape)

        for feature in range(y.shape[3]):
            value = float_hash(self.output_variables[feature], date)
            y[:, :, :, feature] = torch.ones(y[:, :, :, feature].shape) * value

        self.first = False

        return y
