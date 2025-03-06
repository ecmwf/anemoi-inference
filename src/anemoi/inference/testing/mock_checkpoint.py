# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Optional

import yaml
from anemoi.utils.config import DotDict

# "2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitudes", "insolation"
#   0,   1,     2,     3,     4,    5,    6,      7,        8,            9

SIMPLE_METADATA = {
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

HERE = os.path.dirname(__file__)

TEST_CHECKPOINTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))),
    "tests",
    "checkpoints",
)


def mock_load_metadata(path: Optional[str], *, supporting_arrays: bool = True) -> Dict[str, Any]:
    """Load metadata from a YAML file.

    Parameters
    ----------
    path : str, optional
        The path to the checkpoint file.
    supporting_arrays : bool, optional
        Whether to include supporting arrays, by default True.

    Returns
    -------
    dict
        The loaded metadata.
    """

    if path is None:
        metadata = SIMPLE_METADATA
    else:
        path = os.path.basename(path)
        name, _ = os.path.splitext(path)
        for ext in (".yaml", ".json"):
            path = os.path.join(TEST_CHECKPOINTS, f"{name}{ext}")
            if os.path.exists(os.path.join(TEST_CHECKPOINTS, f"{name}{ext}")):
                break

        with open(path) as f:
            metadata = yaml.safe_load(f)

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


def minimum_mock_checkpoint(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create a minimum mock checkpoint from the given metadata.

    Parameters
    ----------
    metadata : dict
        The metadata to create the mock checkpoint from.

    Returns
    -------
    dict
        The minimum mock checkpoint.
    """

    KEEP = {"dataset.variables_metadata", "data_indices.data", "data_indices.model"}

    metadata = deepcopy(metadata)

    def drop(metadata: Dict[str, Any], reference: Dict[str, Any], *path: str) -> None:
        """Recursively drop keys from metadata that are not in the reference.

        Parameters
        ----------
        metadata : dict
            The metadata to modify.
        reference : dict
            The reference metadata to compare against.
        path : str
            The current path in the metadata.
        """

        for k, v in list(metadata.items()):
            key = path + (k,)
            if k not in reference:
                print("Dropping", key, file=sys.stderr)
                del metadata[k]
            else:

                if isinstance(v, dict) and ".".join(key) not in KEEP:
                    drop(metadata[k], reference[k], *path, k)

    drop(metadata, SIMPLE_METADATA)

    variables_metadata = metadata["dataset"]["variables_metadata"]

    for k, v in variables_metadata.items():
        mars = v.get("mars", {})
        for key in ("class", "date", "time", "hdate", "domain", "expver"):
            mars.pop(key, None)

    for k, v in list(metadata["dataset"]["variables_metadata"].items()):
        if k not in SIMPLE_METADATA["dataset"]["variables_metadata"]:
            del metadata["dataset"]["variables_metadata"][k]

    return metadata


if __name__ == "__main__":
    import json
    import sys

    # path= sys.argv[1]
    path = "/Users/mab/git/anemoi-inference/tests/checkpoints/ocean.json"
    print(json.dump(minimum_mock_checkpoint(json.load(open(path)))))
