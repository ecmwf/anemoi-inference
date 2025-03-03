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
from typing import Callable
from unittest.mock import patch

import yaml

HERE = os.path.dirname(__file__)

# Do not include any imports that may load that functoin


def load_metadata(path, supporting_arrays=True) -> None:
    name, _ = os.path.splitext(path)
    with open(os.path.join(HERE, "checkpoints", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def dummy_checkpoints(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("anemoi.utils.checkpoints.load_metadata", load_metadata):
            return func(*args, **kwargs)

    return wrapper


@dummy_checkpoints
def test_checkpoint() -> None:
    from anemoi.inference.checkpoint import Checkpoint

    c = Checkpoint("model.ckpt")
    c.accumulations


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
