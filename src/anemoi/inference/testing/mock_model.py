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
        self.timestep = checkpoint.timestep

        self.first = True
        self.constant_in_time: Dict[int, torch.Tensor] = {}

    def _check(
        self,
        x: torch.Tensor,
        lag: int,
        feature: int,
        date: datetime.datetime,
        step: datetime.timedelta,
        tensor_dates: list[datetime.datetime],
        value: float,
        expect: torch.Tensor,
        title: str,
    ) -> None:
        """Check if the tensor values are as expected.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        lag : int
            The lag index.
        feature : int
            The feature index.
        date : datetime.datetime
            The current date.
        step : datetime.timedelta
            The time step.
        tensor_dates : list of datetime.datetime
            The list of tensor dates.
        value : float
            The expected value.
        expect : torch.Tensor
            The expected tensor.
        title : str
            The title of the check.
        """
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
                "%s: expected %s for %s at lag %s, got %s",
                title,
                value,
                self.input_variables[feature],
                tensor_dates[lag],
                x[:, lag, :, feature][..., 0],
            )
            LOG.error("x: %s", x[:, :, :, feature])

            raise ValueError(
                f"{title}: expected {expect} for {self.input_variables[feature]} at lag {lag}, got {x[:, lag, :, feature][...,0]}"
            )

    def predict_step(
        self, x: torch.Tensor, date: datetime.datetime, step: datetime.timedelta, **kwargs: Any
    ) -> torch.Tensor:
        """Perform a prediction step.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        date : datetime.datetime
            The current date.
        step : datetime.timedelta
            The time step.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        assert x.shape == self.input_shape, f"Expected {self.input_shape}, got {x.shape}"

        # Date of the data
        tensor_dates = [(date - self.timestep) + h for h in self.lagged]

        for lag in range(x.shape[1]):
            for feature in range(x.shape[3]):
                value = float_hash(self.input_variables[feature], tensor_dates[lag])
                expect = torch.Tensor([value])
                variable = self.typed_variables[self.input_variables[feature]]

                if not variable.is_computed_forcing and not variable.is_constant_in_time:
                    # Normal prognostice variable
                    self._check(x, lag, feature, date, step, tensor_dates, value, expect, "Prognostics")
                    continue

                if variable.is_constant_in_time:
                    # Constant in from input, like lsm, only check the first time
                    if self.first:
                        if variable.is_computed_forcing:
                            # Computed forcing, such as cos_latitude. We trust the value to be correct
                            pass
                        else:
                            # Input constant, such as LSM
                            self._check(x, lag, feature, date, step, tensor_dates, value, expect, "Constants")
                        self.constant_in_time[feature] = x[:, lag, :, feature]
                    else:
                        # Check that the value is the same as the first time
                        assert torch.isclose(x[:, lag, :, feature], self.constant_in_time[feature]).all()
                    continue

                if variable.is_computed_forcing:
                    # Computed forcing, such as insolation. We trust the value to be correct
                    continue

                assert False, f"Unknown variable type {variable}"

            self.first = False

        y = torch.zeros(self.output_shape)

        for feature in range(y.shape[3]):
            value = float_hash(self.output_variables[feature], date)
            y[:, :, :, feature] = torch.ones(y[:, :, :, feature].shape) * value

        return y
