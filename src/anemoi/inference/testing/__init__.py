# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import functools
import json
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Union

from anemoi.utils.checkpoints import save_metadata
from anemoi.utils.registry import Registry

testing_registry = Registry(__name__)


def fake_checkpoints(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mock checkpoints for testing.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from .mock_checkpoint import MockRunConfiguration
        from .mock_checkpoint import mock_load_metadata
        from .mock_checkpoint import mock_torch_load

        with (
            patch("anemoi.inference.checkpoint.load_metadata", mock_load_metadata),
            patch("anemoi.inference.provenance.validate_environment", MagicMock()),
            patch("torch.load", mock_torch_load),
            patch("anemoi.inference.metadata.USE_LEGACY", True),
            patch("anemoi.inference.tasks.runner.RunConfiguration", MockRunConfiguration),
        ):
            return func(*args, **kwargs)

    return wrapper


def save_fake_checkpoint(metadata: Union[dict, str, Path], save_path: Path, supporting_arrays: dict = {}) -> None:
    """Create a fake PyTorch checkpoint.

    Parameters
    ----------
    metadata : dict, str, Path
        Metadata dictionary or path to a json file.
    save_path : Path
        The path to the checkpoint file.
    supporting_arrays : dict
        Supporting arrays to be saved in the checkpoint.
    """
    import torch

    from anemoi.inference.testing.mock_model import SimpleMockModel

    if not isinstance(metadata, dict):
        with open(metadata, "r") as f:
            metadata = json.load(f)

    model = SimpleMockModel(metadata, supporting_arrays)

    torch.save(model, save_path)

    save_metadata(
        save_path,
        metadata,
        supporting_arrays=supporting_arrays,
    )


def float_hash(s: str, date: datetime.datetime, accuracy: int = 1_000_000) -> float:
    """Hash a string and date to a float.

    Parameters
    ----------
    s : str
        The string to be hashed.
    date : datetime.datetime
        The date to be hashed.
    accuracy : int, optional
        The accuracy of the hash, by default 1_000_000

    Returns
    -------
    float
        The resulting hash as a float.
    """
    h = s + date.isoformat()
    return float(int.from_bytes(h.encode(), "little") % accuracy) / accuracy


def files_for_tests(name: str) -> str:
    """Get the path to the testing files.

    Parameters
    ----------
    name : str
        The name of the test file.

    Returns
    -------
    str
        The path to the test file.
    """
    # Running in GitHub Actions, the package was installed

    if "GITHUB_WORKSPACE" in os.environ:
        return os.path.join(os.environ["GITHUB_WORKSPACE"], "tests", name)

    # We assume that the test data is in the same directory as the test
    # and this is a development environment

    import anemoi.inference

    bits = os.path.normpath(anemoi.inference.__file__).split(os.path.sep)
    while len(bits) > 3 and (bits[-3], bits[-2], bits[-1]) != ("src", "anemoi", "inference"):
        bits.pop()

    for i in range(3):
        bits.pop()

    bits.append("tests")
    bits.append(name)
    return os.path.sep.join(bits)


class TestingContext:
    """A context for testing plugins."""

    pass
