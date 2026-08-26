# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np
import torch

from anemoi.inference.tensors import TensorHandler


def test_copy_prognostic_fields_to_input_tensor():
    metadata = SimpleNamespace(
        multi_step_input=2,
        multi_step_output=1,
        prognostic_input_mask=np.array([1]),
        prognostic_output_mask=np.array([0]),
    )

    tensor_handler = TensorHandler.__new__(TensorHandler)
    tensor_handler.metadata = metadata
    tensor_handler.trace = False
    tensor_handler._input_kinds = {}
    tensor_handler._input_tensor_by_name = ["force", "prog"]

    input_tensor = torch.tensor(
        [
            [
                [[1.0, 10.0], [1.0, 11.0]],
                [[2.0, 20.0], [2.0, 21.0]],
            ]
        ]
    )

    y_pred = torch.tensor(
        [
            [
                [[30.0, -30.0], [31.0, -31.0]],
            ]
        ]
    )

    check = np.array([False, False])

    result = tensor_handler.copy_prognostic_fields_to_input_tensor(
        input_tensor,
        y_pred,
        check,
    )

    expected = torch.tensor(
        [
            [
                [[2.0, 20.0], [2.0, 21.0]],
                [[1.0, 30.0], [1.0, 31.0]],
            ]
        ]
    )

    assert torch.equal(result, expected)
    assert check.tolist() == [False, True]
    assert tensor_handler._input_kinds["prog"].attributes == {"prognostic": True}
