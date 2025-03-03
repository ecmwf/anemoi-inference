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


def _load_metadata(path, supporting_arrays=True) -> None:
    name, _ = os.path.splitext(path)
    with open(os.path.join(HERE, "checkpoints", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


patch("anemoi.utils.checkpoints.load_metadata", _load_metadata)


def dummy_checkpoints(func: Callable) -> Callable:
    print("dummy_checkpoints", func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("dummy_checkpoints", func)
        with patch("anemoi.utils.checkpoints.load_metadata", _load_metadata):
            from anemoi.utils.checkpoints import load_metadata

            print("dummy_checkpoints", load_metadata)
            return func(*args, **kwargs)

    return wrapper
