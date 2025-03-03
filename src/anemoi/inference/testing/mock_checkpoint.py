# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.utils.config import DotDict

# "2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitudes", "insolation"
#   0,   1,     2,     3,     4,    5,    6,      7,        8,            9


def mock_load_metadata(path: str, *, supporting_arrays=True, name: Any = None) -> dict:
    metadata = {
        "config": {
            "training": {
                "multistep_input": 2,
                "precision": "16-mixed",
            },
            "data": {
                "timestep": "6h",
                "forcing": ["lsm", "z"],
                "diagnostic": ["tcc", "tp"],
            },
        },
        "data_indices": {
            "data": {
                "input": {"full": [0, 1, 2, 3, 4, 5, 8, 9], "forcing": [4, 5, 8, 9], "diagnostic": [6, 7]},
                "output": {"full": [0, 1, 2, 3, 6, 7]},
            },
            "model": {
                "input": {"full": [0, 1, 2, 3, 4, 5, 6, 7], "forcing": [4, 5, 6, 7], "prognostic": [0, 1, 2, 3]},
                "output": {"full": [0, 1, 2, 3, 4, 5], "prognostic": [0, 1, 2, 3]},
            },
        },
        "dataset": {
            "variables": ["2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitude", "insolation"],
            "shape": [365, 10, 1, 40320],
            "variables_metadata": {
                "2t": {},
                "10u": {},
                "10v": {},
                "msl": {},
                "lsm": {"constant_in_time": True},
                "z": {"constant_in_time": True},
                "tcc": {},
                "tp": {"accumulated": True},
                "cos_latitude": {"computed_forcing": True, "constant_in_time": True},
                "insolation": {"computed_forcing": True, "constant_in_time": False},
            },
        },
    }

    arrays = {}

    if supporting_arrays:
        return metadata, arrays
    return metadata


def mock_torch_load(path: str, map_location: Any, weights_only: bool) -> Any:
    import torch

    assert weights_only is False, "Not implemented"

    class MockModel(torch.nn.Module):
        def __init__(self, medatada, supporting_arrays):
            super().__init__()
            metadata = DotDict(medatada)

            self.features_in = len(metadata.data_indices.model.input.full)
            self.features_out = len(metadata.data_indices.model.output.full)
            self.roll_window = metadata.config.training.multistep_input
            self.grid_size = metadata.dataset.shape[-1]

            self.input_shape = (1, self.roll_window, self.grid_size, self.features_in)
            self.output_shape = (1, 1, self.grid_size, self.features_out)

        def forward(self, x):
            assert False, "Not implemented"

        def predict_step(self, x):

            assert x.shape == self.input_shape, f"Expected {self.input_shape}, got {x.shape}"

            y = torch.zeros(self.output_shape)

            return y

    return MockModel(*mock_load_metadata(path))
