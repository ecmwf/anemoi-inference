# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for runners/testing.py: SteadyStateTensorHandler and NoModelMixing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from anemoi.inference.runners.testing import NoModelMixing
from anemoi.inference.runners.testing import SteadyStateTensorHandler
from anemoi.inference.tensors import TensorHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(dynamic_masks):
    """Return a bare SteadyStateTensorHandler with mocked forcings providers."""
    handler = SteadyStateTensorHandler.__new__(SteadyStateTensorHandler)
    handler.dynamic_forcings_providers = [SimpleNamespace(mask=m) for m in dynamic_masks]
    return handler


# ---------------------------------------------------------------------------
# SteadyStateTensorHandler
# ---------------------------------------------------------------------------


class TestSteadyStateTensorHandler:
    def test_is_subclass_of_tensor_handler(self):
        assert issubclass(SteadyStateTensorHandler, TensorHandler)

    def test_check_marked_true_for_dynamic_forcings(self):
        """Dynamic-forcing slots must be marked as filled in the check array."""
        mask = np.array([2, 4])  # columns 2 and 4 are dynamic forcings
        handler = _make_handler([mask])

        n_vars = 6
        tensor = torch.zeros(1, 1, 10, n_vars)
        check = np.zeros(n_vars, dtype=bool)

        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert check[2] and check[4]
        assert not check[0] and not check[1] and not check[3] and not check[5]

    def test_tensor_values_unchanged(self):
        """The input tensor must NOT be modified — forcings are frozen."""
        mask = np.array([1, 3])
        handler = _make_handler([mask])

        original = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        tensor = original.clone()
        check = np.zeros(6, dtype=bool)

        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert torch.equal(tensor, original), "tensor was modified — forcings should be frozen"

    def test_multiple_providers_all_marked(self):
        """All providers' masks must be reflected in the check array."""
        handler = _make_handler([np.array([0]), np.array([3, 5])])

        check = np.zeros(6, dtype=bool)
        tensor = torch.zeros(1, 1, 4, 6)
        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert check[0] and check[3] and check[5]
        assert not check[1] and not check[2] and not check[4]

    def test_no_providers_leaves_check_unchanged(self):
        """With no dynamic-forcings providers the check array stays all False."""
        handler = _make_handler([])

        check = np.zeros(4, dtype=bool)
        tensor = torch.zeros(1, 1, 4, 4)
        handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)

        assert not check.any()

    def test_returns_same_tensor_object(self):
        """The method must return the (same) input tensor."""
        handler = _make_handler([np.array([0])])
        tensor = torch.zeros(1, 1, 4, 3)
        check = np.zeros(3, dtype=bool)
        returned = handler.add_dynamic_forcings_to_input_tensor(tensor, state={}, dates=[], check=check)
        assert returned is tensor


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

        input_tensor = torch.zeros(1, 2, 40, 100)  # batch, multi_step_input, gridpoints, n_input_vars
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


# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------


def test_steady_state_runner_is_registered():
    from anemoi.inference.runners import runner_registry

    assert runner_registry.is_registered("steady-state"), "'steady-state' runner not found in registry"


def test_no_model_runner_is_registered():
    from anemoi.inference.runners import runner_registry

    assert runner_registry.is_registered("no-model"), "'no-model' runner not found in registry"
