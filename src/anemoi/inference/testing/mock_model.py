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

from anemoi.inference.metadata import Metadata
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
class SimpleMetadata:
    """Some Metadata attributes are not serialisable, so we store only a subset of metadata in here, for use in the mock model or testing.

    Attributes are sourced directly from Metadata.
    """

    output_shape: tuple[int, ...]
    variable_to_output_tensor_index: dict[str, int]
    output_tensor_index_to_variable: dict[int, str]
    prognostic_variables: list[str]
    diagnostic_variables: list[str]
    prognostic_input_indices: list[int]
    prognostic_output_indices: list[int]
    diagnostic_output_indices: list[int]

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

        variable_to_input_tensor_index = dict(metadata.variable_to_input_tensor_index)
        variable_to_output_tensor_index = dict(metadata.variable_to_output_tensor_index)

        return cls(
            output_shape=tuple(metadata.output_shape),
            variable_to_output_tensor_index=variable_to_output_tensor_index,
            output_tensor_index_to_variable=dict(metadata.output_tensor_index_to_variable),
            prognostic_variables=prognostic_variables,
            diagnostic_variables=diagnostic_variables,
            prognostic_input_indices=[variable_to_input_tensor_index[v] for v in prognostic_variables],
            prognostic_output_indices=[variable_to_output_tensor_index[v] for v in prognostic_variables],
            diagnostic_output_indices=[variable_to_output_tensor_index[v] for v in diagnostic_variables],
        )


class SimpleMockModel(torch.nn.Module):
    """Simple mock model with configuration options for testing.

    Behaviour is determined by the `prognostic_output` argument: `PASSTHROUGH` copies prognostics from input to output and
    fills the diagnostics with ones, `ONES` ignores the input, and `HASH` fills every output variable with `float_hash(variable, date)`.

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
    ) -> None:
        super().__init__()

        self.prognostic_output = VariableOutputSetting(prognostic_output)
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
