# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for runners/testing.py: NoModelMixing / NoModelRunner.

"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from anemoi.inference.runners.testing import NoModelMixing

# ---------------------------------------------------------------------------
# NoModelMixing
# ---------------------------------------------------------------------------


def _make_no_model_runner(output_shape):
    """Return a NoModelMixing instance whose checkpoint reports the given output shape."""
    metadata = SimpleNamespace(output_shape=output_shape)
    checkpoint = MagicMock()
    checkpoint.multi_dataset_metadata = {"data": metadata}

    runner = NoModelMixing.__new__(NoModelMixing)
    # NoModelMixing is a mixin; in production it inherits Runner.checkpoint.
    # Here we mock it directly as an attribute.
    runner.checkpoint = checkpoint
    return runner


class TestNoModelMixing:
    def test_output_shape_matches_metadata(self):
        """predict_step must return a tensor matching metadata.output_shape."""
        output_shape = (1, 1, 1, 40, 88)  # batch, time, ensemble, gridpoints, vars
        runner = _make_no_model_runner(output_shape)
        model = runner.model

        input_tensor = torch.zeros(1, 2, 40, 100)
        result = model.predict_step({"data": input_tensor})

        assert result["data"].shape == torch.Size(output_shape)

    def test_output_filled_with_ones(self):
        """predict_step must return all-ones (sentinel value for no-model)."""
        output_shape = (1, 1, 1, 10, 5)
        runner = _make_no_model_runner(output_shape)
        model = runner.model

        input_tensor = torch.zeros(1, 1, 10, 8)
        result = model.predict_step({"data": input_tensor})

        assert (result["data"] == 1).all()

    def test_legacy_single_tensor_input(self):
        """predict_step must handle a plain tensor (legacy single-dataset path)."""
        output_shape = (1, 1, 1, 10, 4)
        runner = _make_no_model_runner(output_shape)
        model = runner.model

        input_tensor = torch.zeros(1, 1, 10, 6)
        result = model.predict_step(input_tensor)  # not a dict

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size(output_shape)

    def test_dtype_and_device_preserved(self):
        """Output dtype and device must match the input tensor."""
        output_shape = (1, 1, 1, 8, 3)
        runner = _make_no_model_runner(output_shape)
        model = runner.model

        input_tensor = torch.zeros(1, 1, 8, 5, dtype=torch.float64)
        result = model.predict_step({"data": input_tensor})

        assert result["data"].dtype == torch.float64


