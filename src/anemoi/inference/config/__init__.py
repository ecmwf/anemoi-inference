# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Union

import yaml
from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Configuration(BaseModel):

    class Config:
        extra = "forbid"

    @classmethod
    def load(
        cls,
        path: Union[str, dict],
        overrides: Union[List[str], List[dict], str, dict] = [],
        defaults: Optional[Union[str, List[str], dict]] = None,
    ) -> Configuration:

        config = {}

        # Set default values
        if defaults is not None:
            if not isinstance(defaults, list):
                defaults = [defaults]
            for d in defaults:
                if isinstance(d, str):
                    with open(d) as f:
                        d = yaml.safe_load(f)
                config.update(d)

        # Load the configuration
        if isinstance(path, dict):
            config = deepcopy(path)
        else:
            with open(path) as f:
                config.update(yaml.safe_load(f))

        # Apply overrides
        if not isinstance(overrides, list):
            overrides = [overrides]

        for override in overrides:
            if isinstance(override, dict):
                cls._merge_configs(config, override)
            else:
                path = config
                key, value = override.split("=")
                keys = key.split(".")
                for key in keys[:-1]:
                    path = path.setdefault(key, {})
                path[keys[-1]] = value

        # Validate the configuration
        config = cls(**config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config

    @classmethod
    def _merge_configs(cls, a: dict, b: dict) -> None:
        for key, value in b.items():
            if key in a and isinstance(a[key], dict) and isinstance(value, dict):
                cls._merge_configs(a[key], value)
            else:
                a[key] = value
