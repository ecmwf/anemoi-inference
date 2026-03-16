# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from dataclasses import dataclass
from typing import Any

import torch
from anemoi.utils.config import DotDict

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.checkpoint import get_multi_dataset_metadata
from anemoi.inference.config import LOG
from anemoi.inference.metadata import MetadataFactory
from anemoi.inference.testing import float_hash


@dataclass
class SimpleMetadata:
    """Some Metadata attributes are not serialisable, so we store only the necessary information in here."""

    multi_step_output: int
    prognostic_variables: list[str]
    variable_to_input_index: dict[str, int]
    output_index_to_variable: dict[int, str]
    variable_to_output_index: dict[str, int]
    features_out: int
    is_multi_out: bool
    prognostic_input_indices: list[int]
    prognostic_output_indices: list[int]


class SimpleMockModel(torch.nn.Module):
    """Simple mock model that copies prognostics from input to output. Diagnostics are filled with ones.
    In the case of multi-step input, only the last input step is copied to the output.
    In the case of multi-step output, the same output is repeated for each output step.

    This model has support for legacy models and newer multi-dataset models.
    When loaded with legacy metadata, the `predict_step` will take a single tensor and output a single output tensor.
    When loaded with multi-dataset metadata, the `predict_step` will take a dict of tensors and output a dict of tensors,
    with the dataset names, as defined in the metadata, as keys.
    """

    def __init__(self, raw_metadata: dict, supporting_arrays: dict) -> None:
        super().__init__()
        multi_metadata = get_multi_dataset_metadata(raw_metadata, supporting_arrays)

        self.metadata: dict[str, SimpleMetadata] = {}

        for dataset, metadata in multi_metadata.items():
            prognostic_variables = metadata.select_variables(include=["prognostic"], has_mars_requests=False)
            variable_to_input_index = dict(metadata.variable_to_input_tensor_index)
            output_index_to_variable = dict(metadata.output_tensor_index_to_variable)
            variable_to_output_index = dict(metadata.variable_to_output_tensor_index)

            self.metadata[dataset] = SimpleMetadata(
                multi_step_output=metadata.multi_step_output,
                prognostic_variables=prognostic_variables,
                variable_to_input_index=variable_to_input_index,
                output_index_to_variable=output_index_to_variable,
                variable_to_output_index=variable_to_output_index,
                prognostic_input_indices=[variable_to_input_index[var] for var in prognostic_variables],
                prognostic_output_indices=[variable_to_output_index[var] for var in prognostic_variables],
                features_out=len(output_index_to_variable),
                is_multi_out=hasattr(metadata._config_training, "multistep_output"),
            )

    def predict_step(
        self,
        input_tensor: torch.Tensor | dict[str, torch.Tensor],
        date: datetime.datetime = None,
        step: datetime.timedelta = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not isinstance(input_tensor, dict):
            # legacy model: the runner will pass a single tensor if it detects a legacy checkpoint, we must return a tensor
            # we could hardcode "data" here, but better to rely on the default key set by the metadata factory
            assert len(self.metadata) == 1, "Expected a single dataset in the metadata for legacy model"
            default_key = next(iter(self.metadata.keys()))
            return self._predict_step({default_key: input_tensor})[default_key]

        # multi-dataset model
        return self._predict_step(input_tensor)

    def _predict_step(self, input_tensor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}

        for dataset, metadata in self.metadata.items():
            output_shape = (
                1,  # batch
                metadata.multi_step_output,  # time
                1,  # ensemble
                input_tensor[dataset].shape[2],  # values
                metadata.features_out,  # variables
            )

            # for legacy models without multi-step output, we also output a single time step
            # this is not technically required because the runner has a switch to unsqueeze the time dimension when it detects this
            # this ensures the switch is triggered and tested
            # TODO: if all test checkpoints are ever updated to multi-step output, this can be removed
            if not metadata.is_multi_out:
                output_shape = (1, 1, input_tensor[dataset].shape[2], metadata.features_out)

            output_tensor = torch.ones(
                *output_shape, dtype=input_tensor[dataset].dtype, device=input_tensor[dataset].device
            )

            # copy prognostic input variables of the last time step to output tensor
            if metadata.prognostic_input_indices:
                output_tensor[..., metadata.prognostic_output_indices] = input_tensor[dataset][
                    :, -1, :, metadata.prognostic_input_indices
                ]

            outputs[dataset] = output_tensor

        return outputs


class LegacyMockModel(torch.nn.Module):
    """Mock model with internal sanity checks. Assumes the input comes from the `dummy` input source."""

    def __init__(self, metadata: dict[str, Any], supporting_arrays: dict[str, Any]) -> None:
        super().__init__()
        metadata = DotDict(metadata)
        self.supporting_arrays = supporting_arrays

        checkpoint = Checkpoint(MetadataFactory(metadata))

        self.features_in = len(checkpoint.variable_to_input_tensor_index)
        self.features_out = len(checkpoint.output_tensor_index_to_variable)
        self.roll_window = checkpoint.multi_step_input
        self.grid_size = checkpoint._metadata.number_of_grid_points
        self.multi_step_output = checkpoint.multi_step_output

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
