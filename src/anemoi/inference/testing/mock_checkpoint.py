# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Dict

from anemoi.utils.config import DotDict

# "2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitudes", "insolation"
#   0,   1,     2,     3,     4,    5,    6,      7,        8,            9


def mock_load_metadata(path: str, *, supporting_arrays: bool = True, name: Any = None) -> Dict[str, Any]:
    """Load mock metadata for testing purposes.

    Parameters
    ----------
    path : str
        The path to the metadata file.
    supporting_arrays : bool, optional
        Whether to include supporting arrays in the output, by default True.
    name : Any, optional
        An optional name parameter, by default None.

    Returns
    -------
    dict
        The mock metadata.
    dict, optional
        The supporting arrays if `supporting_arrays` is True.
    """
    metadata: Dict[str, Any] = {
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
                "2t": {"mars": {"param": "2t", "levtype": "sfc"}},
                "10u": {"mars": {"param": "10u", "levtype": "sfc"}},
                "10v": {"mars": {"param": "10v", "levtype": "sfc"}},
                "msl": {"mars": {"param": "msl", "levtype": "sfc"}},
                "lsm": {"constant_in_time": True, "mars": {"param": "lsm", "levtype": "sfc"}},
                "z": {"constant_in_time": True, "mars": {"param": "z", "levtype": "sfc"}},
                "tcc": {"mars": {"param": "tcc", "levtype": "sfc"}},
                "tp": {"accumulated": True, "mars": {"param": "tp", "levtype": "sfc"}},
                "cos_latitude": {"computed_forcing": True, "constant_in_time": True},
                "insolation": {"computed_forcing": True, "constant_in_time": False},
            },
            "data_request": {"grid": "O96"},
        },
    }

    arrays: Dict[str, Any] = {}

    if supporting_arrays:
        return metadata, arrays
    return metadata


def mock_torch_load(path: str, map_location: Any, weights_only: bool) -> Any:
    """Load a mock torch model for testing purposes.

    Parameters
    ----------
    path : str
        The path to the model file.
    map_location : Any
        The device on which to load the model.
    weights_only : bool
        Whether to load only the weights.

    Returns
    -------
    Any
        The mock torch model.
    """
    import torch

    assert weights_only is False, "Not implemented"

    class MockModel(torch.nn.Module):
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

        def predict_step(self, x: torch.Tensor) -> torch.Tensor:
            """Perform a prediction step.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor.

            Returns
            -------
            torch.Tensor
                The output tensor.
            """
            assert x.shape == self.input_shape, f"Expected {self.input_shape}, got {x.shape}"

            y = torch.zeros(self.output_shape)

            return y

    return MockModel(*mock_load_metadata(path))
