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

import yaml

from anemoi.inference.testing import files_for_tests

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


def mock_load_metadata(path: str | None, *, supporting_arrays: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
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
        if not os.path.isabs(path):
            path = files_for_tests(path)
        name, _ = os.path.splitext(path)
        for ext in (".yaml", ".json"):
            path = f"{name}{ext}"
            if os.path.exists(path):
                break

        with open(path) as f:
            metadata = yaml.safe_load(f)

    arrays: dict[str, Any] = {}

    if supporting_arrays:
        return metadata, arrays

    return metadata


def minimum_mock_checkpoint(metadata: dict[str, Any]) -> dict[str, Any]:
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

    def drop(metadata: dict[str, Any], reference: dict[str, Any], *path: str) -> None:
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

        if ".".join(path) in KEEP:
            return

        for k, v in list(metadata.items()):
            if k not in reference:
                del metadata[k]
            else:
                if isinstance(v, dict):
                    drop(metadata[k], reference[k], *path, k)

    drop(metadata, SIMPLE_METADATA)

    variables_metadata = metadata["dataset"]["variables_metadata"]

    for k, v in variables_metadata.items():
        mars = v.get("mars", {})
        for key in ("class", "date", "time", "hdate", "domain", "expver"):
            mars.pop(key, None)

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
    from .mock_model import MockModel

    assert weights_only is False, "Not implemented"

    metadata, arrays = mock_load_metadata(path)

    return MockModel(metadata, arrays)


class MockRunConfiguration:
    """Mock class for loading run configurations from `files_for_tests`."""

    @classmethod
    def load(cls, path: str, *args, **kwargs) -> dict[str, Any]:
        """Load a run configuration from a given path.

        Parameters
        ----------
        path : str
            The path to the run configuration file.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            The loaded run configuration.
        """
        from anemoi.inference.config.run import RunConfiguration

        if not os.path.isabs(path):
            path = files_for_tests(path)

        return RunConfiguration.load(path, *args, **kwargs)
