# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from anemoi.utils.config import DotDict

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.config import LOG
from anemoi.inference.metadata import Metadata
from anemoi.inference.metadata import MetadataFactory
from anemoi.inference.testing import float_hash


class VariableOutputSetting(StrEnum):
    """Setting which describes the variable behaviour of the mock model.

    Attributes
    ----------
    ONES : str
        Ignore the input and return ones.
    PASSTHROUGH : str
        Copy the prognostic variables of the last input step into the output.
    HASH : str
        Fill every output variable with `float_hash(variable, date)`.
    """

    ONES = "ones"
    PASSTHROUGH = "passthrough"
    HASH = "hash"


@dataclass(frozen=True)
class VariableFlags:
    """Serialisable and minimal summary of a variable's type.

    Attributes
    ----------
    is_computed_forcing : bool
        Whether the variable is computed by the runner rather than read from the input.
    is_constant_in_time : bool
        Whether the variable holds the same value at every date.
    is_diagnostic : bool
        Whether the variable is produced by the model rather than fed to it.
    """

    is_computed_forcing: bool
    is_constant_in_time: bool
    is_diagnostic: bool


@dataclass(frozen=True)
class SimpleMetadata:
    """Some Metadata attributes are not serialisable, so we store only a subset of metadata in here, for use in the mock model or testing.

    Attributes are sourced directly from Metadata.
    """

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    lagged: list[datetime.timedelta]
    timestep: datetime.timedelta
    variable_to_input_tensor_index: dict[str, int]
    input_tensor_index_to_variable: dict[int, str]
    variable_to_output_tensor_index: dict[str, int]
    output_tensor_index_to_variable: dict[int, str]
    prognostic_variables: list[str]
    diagnostic_variables: list[str]
    prognostic_input_indices: list[int]
    prognostic_output_indices: list[int]
    diagnostic_output_indices: list[int]
    variable_flags: dict[str, VariableFlags]

    @classmethod
    def _from_metadata(cls, metadata: Metadata) -> "SimpleMetadata":
        """Reduce the metadata of one dataset to plain, picklable data.

        Parameters
        ----------
        metadata : Metadata
            The metadata of a single dataset.

        Returns
        -------
        SimpleMetadata
            The reduced metadata.
        """
        prognostic_variables = metadata.select_variables(include=["prognostic"], has_mars_requests=False)
        diagnostic_variables = metadata.select_variables(include=["diagnostic"], has_mars_requests=False)
        diagnostics = set(diagnostic_variables)

        variable_to_input_tensor_index = dict(metadata.variable_to_input_tensor_index)
        variable_to_output_tensor_index = dict(metadata.variable_to_output_tensor_index)

        return cls(
            input_shape=tuple(metadata.input_shape),
            output_shape=tuple(metadata.output_shape),
            lagged=list(metadata.lagged),
            timestep=metadata.timestep,
            variable_to_input_tensor_index=variable_to_input_tensor_index,
            input_tensor_index_to_variable={v: k for k, v in variable_to_input_tensor_index.items()},
            variable_to_output_tensor_index=variable_to_output_tensor_index,
            output_tensor_index_to_variable=dict(metadata.output_tensor_index_to_variable),
            prognostic_variables=prognostic_variables,
            diagnostic_variables=diagnostic_variables,
            prognostic_input_indices=[variable_to_input_tensor_index[v] for v in prognostic_variables],
            prognostic_output_indices=[variable_to_output_tensor_index[v] for v in prognostic_variables],
            diagnostic_output_indices=[variable_to_output_tensor_index[v] for v in diagnostic_variables],
            variable_flags={
                name: VariableFlags(
                    is_computed_forcing=bool(variable.is_computed_forcing),
                    is_constant_in_time=bool(variable.is_constant_in_time),
                    is_diagnostic=name in diagnostics,
                )
                for name, variable in metadata.typed_variables.items()
            },
        )


class SimpleMockModel(torch.nn.Module):
    """Simple mock model with configuration options for testing.

    Behavior is determined by the `variable_output` argument: `PASSTHROUGH` copies prognostics from input to output and fills the diagnostics with ones,
    `ONES` ignores the input, and `HASH` fills every output variable with `float_hash(variable, date)`.
    In the case of multi-step input, only the last input step is copied to the output.
    In the case of multi-step output, the same output is repeated for each output step.

    This model has support for legacy models and newer multi-dataset models.
    When loaded with legacy metadata, the `predict_step` will take a single tensor and output a single output tensor.
    When loaded with multi-dataset metadata, the `predict_step` will take a dict of tensors and output a dict of tensors,
    with the dataset names, as defined in the metadata, as keys.
    """

    def __init__(
        self,
        multi_metadata: Mapping[str, Metadata],
        *,
        prognostic_output: str = VariableOutputSetting.PASSTHROUGH,
        validate: bool = False,
    ) -> None:
        super().__init__()

        self.prognostic_output = VariableOutputSetting(prognostic_output)
        self.validate = validate
        self.metadata: dict[str, SimpleMetadata] = {
            dataset: SimpleMetadata._from_metadata(metadata) for dataset, metadata in multi_metadata.items()
        }

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
            return self._predict_step({default_key: input_tensor}, date)[default_key]

        # multi-dataset model
        return self._predict_step(input_tensor, date)

    def _predict_step(
        self,
        input_tensor: dict[str, torch.Tensor],
        date: datetime.datetime | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}

        for dataset, metadata in self.metadata.items():
            output_tensor = torch.ones(
                *metadata.output_shape,
                dtype=input_tensor[dataset].dtype,
                device=input_tensor[dataset].device,
            )

            if self.prognostic_output == VariableOutputSetting.HASH:
                if date is None:
                    raise ValueError(
                        f"The {VariableOutputSetting.HASH.value!r} output rule needs a date, none was passed"
                    )
                for index, variable in metadata.output_tensor_index_to_variable.items():
                    output_tensor[..., index] = float_hash(variable, date)

            # copy prognostic input variables of the last time step to output tensor, leaving diagnostics at one
            elif self.prognostic_output == VariableOutputSetting.PASSTHROUGH and metadata.prognostic_input_indices:
                output_tensor[..., metadata.prognostic_output_indices] = input_tensor[dataset][
                    :, -1:, :, metadata.prognostic_input_indices
                ]

            outputs[dataset] = output_tensor

        return outputs


class LegacyMockModel(torch.nn.Module):
    """Mock model with internal sanity checks. Assumes the input comes from the `dummy` input source."""

    def __init__(self, raw_metadata: dict[str, Any], supporting_arrays: dict[str, Any]) -> None:
        super().__init__()
        raw_metadata = DotDict(raw_metadata)
        self.supporting_arrays = supporting_arrays

        # this model is only used in unit tests with legacy checkpoints
        # we don't have to account for multi-datasets
        # TODO: re-assess if the unit tests get updated to multi-dataset checkpoints
        metadata = Checkpoint(MetadataFactory(raw_metadata))._metadata

        self.input_shape = metadata.input_shape
        self.output_shape = metadata.output_shape

        self.variable_to_input_index = dict(metadata.variable_to_input_tensor_index)
        self.input_index_to_variable = {v: k for k, v in self.variable_to_input_index.items()}

        self.output_index_to_variable = dict(metadata.output_tensor_index_to_variable)
        self.variable_to_output_index = {v: k for k, v in self.output_index_to_variable.items()}

        self.lagged = metadata.lagged
        self.typed_variables = metadata.typed_variables
        self.prognostic_variables = metadata.select_variables(include=["prognostic"], has_mars_requests=False)
        self.diagnostic_variables = metadata.select_variables(include=["diagnostic"], has_mars_requests=False)
        self.timestep = metadata.timestep

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
