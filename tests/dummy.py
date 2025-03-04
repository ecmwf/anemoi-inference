# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from functools import wraps
from typing import Any
from typing import Callable
from unittest.mock import patch

import yaml

HERE = os.path.dirname(__file__)

# Do not include any imports that may load that functoin


def _load_metadata(path: str, supporting_arrays: bool = True) -> Any:
    """Load metadata from a YAML file.

    Parameters
    ----------
    path : str
        The path to the metadata file.
    supporting_arrays : bool, optional
        Whether to include supporting arrays, by default True

    Returns
    -------
    Any
        The loaded metadata.
    """
    name, _ = os.path.splitext(path)
    with open(os.path.join(HERE, "checkpoints", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


patch("anemoi.utils.checkpoints.load_metadata", _load_metadata)


def dummy_checkpoints(func: Callable) -> Callable:
    """Decorator to patch the load_metadata function with a dummy implementation.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:

        with patch("anemoi.utils.checkpoints.load_metadata", _load_metadata):
            from anemoi.utils.checkpoints import load_metadata

            print("dummy_checkpoints", load_metadata)
            return func(*args, **kwargs)

    return wrapper
