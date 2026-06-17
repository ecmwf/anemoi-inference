# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import torch

from anemoi.inference.runners.downscaling import DownscalingRunner


class RecordingModel:
    def __init__(self, output_tensor):
        self.output_tensor = output_tensor
        self.called_with = None

    def predict_step(self, low_res_tensor, high_res_tensor, *, extra_args, **kwargs):
        self.called_with = (low_res_tensor, high_res_tensor, extra_args, kwargs)
        return self.output_tensor


def test_predict_step_accepts_four_channel_low_res_and_output_tensors():
    runner = object.__new__(DownscalingRunner)
    runner.config = SimpleNamespace(development_hacks={})

    low_res_tensor = torch.zeros(1, 1, 8, 4)
    high_res_tensor = torch.zeros(1, 1, 16, 10)
    output_tensor = torch.ones(1, 1, 16, 4)
    runner._prepare_high_res_input_tensor = lambda input_date: high_res_tensor
    model = RecordingModel(output_tensor)

    result = DownscalingRunner.predict_step(
        runner,
        model,
        low_res_tensor,
        date=datetime(2025, 9, 26),
        step=timedelta(hours=24),
    )

    assert result is output_tensor
    assert model.called_with == (low_res_tensor, high_res_tensor, {}, {})
