# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
from typing import Any
from typing import Callable


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

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from unittest.mock import patch

        from .mock_checkpoint import mock_load_metadata
        from .mock_checkpoint import mock_torch_load

        with (
            patch("anemoi.inference.checkpoint.load_metadata", mock_load_metadata),
            patch("torch.load", mock_torch_load),
            patch("anemoi.inference.metadata.USE_LEGACY", True),
        ):
            return func(*args, **kwargs)

    return wrapper


def float_hash(s: str, date: datetime.datetime, offset: int = 0, accuracy=1_000_000) -> float:
    """Hash a string and date to a float."""
    h = s + date.isoformat()
    return float(int.from_bytes(h.encode(), "little") % accuracy) / accuracy + offset
