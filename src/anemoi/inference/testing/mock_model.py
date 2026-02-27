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

import torch
from anemoi.utils.config import DotDict

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.config import LOG
from anemoi.inference.metadata import MetadataFactory
from anemoi.inference.testing import float_hash


class MockModelBase(torch.nn.Module):
    """Mock model base class for testing."""

    def __init__(self, metadata: dict[str, Any], supporting_arrays: dict[str, Any]) -> None:
        super().__init__()
        metadata = DotDict(metadata)
        self.supporting_arrays = supporting_arrays

        checkpoint = Checkpoint(MetadataFactory(metadata))

        self.features_in = len(checkpoint.variable_to_input_tensor_index)
        self.features_out = len(checkpoint.output_tensor_index_to_variable)
        self.roll_window = checkpoint.multi_step_input
        self.grid_size = checkpoint.number_of_grid_points
        self.multi_step_output = checkpoint.multi_step_output

        self._is_multi_out = hasattr(metadata.config.training, "multistep_output")

        self.input_shape = (1, self.roll_window, self.grid_size, self.features_in)
        self.output_shape = (1, self.multi_step_output, self.grid_size, self.features_out)

        self.variable_to_input_index = dict(checkpoint.variable_to_input_tensor_index)
        self.input_index_to_variable = {v: k for k, v in self.variable_to_input_index.items()}

        self.output_index_to_variable = dict(checkpoint.output_tensor_index_to_variable)
        self.variable_to_output_index = {v: k for k, v in self.output_index_to_variable.items()}

        self.lagged = checkpoint.lagged
        self.typed_variables = checkpoint.typed_variables
        self.prognostic_variables = checkpoint.select_variables(include=["prognostic"], has_mars_requests=False)
        self.diagnostic_variables = checkpoint.select_variables(include=["diagnostic"], has_mars_requests=False)
        self.timestep = checkpoint.timestep

        self.first = True
        self.constant_in_time: dict[int, torch.Tensor] = {}

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

        raise NotImplementedError("This method should be implemented by subclasses.")


class SimpleMockModel(MockModelBase):
    """Simple mock model that copies prognostics from input to output. Diagnostics are filled with ones.
    In the case of multi-step input, the last time step is copied.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prognostic_input_indices = [self.variable_to_input_index[var] for var in self.prognostic_variables]
        self.prognostic_output_indices = [self.variable_to_output_index[var] for var in self.prognostic_variables]

    # NOTE: This is a temporary hack to support the single dataset case for multi-dataset mock models
    # TODO: change when we have a proper multi-dataset (mock) model
    def predict_step(
        self, x: torch.Tensor, date: datetime.datetime = None, step: datetime.timedelta = None, **kwargs: Any
    ) -> torch.Tensor:
        if isinstance(x, dict):
            assert "data" in x, "Expected input to be a dict with a 'data' key"
            assert len(x.keys()) == 1, "Expected input dict to only contain a 'data' key"
            return {"data": self._predict_step(x["data"], date, step, **kwargs)}
        return self._predict_step(x, date, step, **kwargs)

    def _predict_step(
        self, x: torch.Tensor, date: datetime.datetime, step: datetime.timedelta, **kwargs: Any
    ) -> torch.Tensor:
        output_shape = (
            1,  # batch
            self.multi_step_output,  # output_times
            1,  # time
            x.shape[2],  # gridpoints
            self.features_out,  # variables
        )
        # TODO remove this when all tests are updated to use multi-step output
        if not self._is_multi_out:
            output_shape = (1, 1, x.shape[2], self.features_out)  # for backwards compatibility
        y = torch.ones(*output_shape, dtype=x.dtype, device=x.device)

        # copy prognostic input variables of the last time step to output tensor
        if self.prognostic_input_indices:
            y[..., self.prognostic_output_indices] = x[:, -1, :, self.prognostic_input_indices]

        return y


class MockModel(MockModelBase):
    """Mock model with internal sanity checks. Assumes the input comes from the `dummy` input source."""

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
                self.input_index_to_variable[feature],
                lag,
                tensor_dates[lag],
            )
            LOG.error("Date: %s, Step: %s, Tensor dates: %s", date, step, tensor_dates)
            LOG.error(
                "%s: expected %s for %s at lag %s, got %s",
                title,
                value,
                self.input_index_to_variable[feature],
                tensor_dates[lag],
                x[:, lag, :, feature][..., 0],
            )
            LOG.error("x: %s", x[:, :, :, feature])

            raise ValueError(
                f"{title}: expected {expect} for {self.input_index_to_variable[feature]} at lag {lag}, got {x[:, lag, :, feature][...,0]}"
            )

    def predict_step(
        self, x: torch.Tensor, date: datetime.datetime, step: datetime.timedelta, **kwargs: Any
    ) -> torch.Tensor:
        assert x.shape == self.input_shape, f"Expected {self.input_shape}, got {x.shape}"

        # Date of the data
        tensor_dates = [(date - self.timestep) + h for h in self.lagged]

        for lag in range(x.shape[1]):
            for feature in range(x.shape[3]):
                value = float_hash(self.input_index_to_variable[feature], tensor_dates[lag])
                expect = torch.Tensor([value])
                variable = self.typed_variables[self.input_index_to_variable[feature]]

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
            value = float_hash(self.output_index_to_variable[feature], date)
            y[:, :, :, feature] = torch.ones(y[:, :, :, feature].shape) * value

        return y
