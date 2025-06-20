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
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

from earthkit.data.utils.dates import to_datetime
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

LOG = logging.getLogger(__name__)

T = TypeVar("T", bound="Configuration")


class Configuration(BaseModel):
    """Configuration class."""

    model_config = ConfigDict(extra="forbid")

    date: Union[datetime, None] = None
    """The starting date for the forecast. If not provided, the date will depend on the selected Input object. If a string, it is parsed by :func:`earthkit.data.utils.dates`."""

    @field_validator("date", mode="before")
    @classmethod
    def to_datetime(cls, date: Union[str, int, datetime, None]) -> Optional[datetime]:
        if date is not None:
            return to_datetime(date)

    @classmethod
    def load(
        cls: Type[T],
        path: Union[str, Dict[str, Any]],
        overrides: Union[List[str], List[dict], str, dict] = [],
        defaults: Optional[Union[str, List[str], dict]] = None,
    ) -> T:
        """Load the configuration.

        Parameters
        ----------
        path : Union[str, dict]
            Path to the configuration file or a dictionary containing the configuration.
        overrides : Union[List[str], List[dict], str, dict], optional
            List of overrides to apply to the configuration, by default [].
        defaults : Optional[Union[str, List[str], dict]], optional
            Default values to set in the configuration, by default None.

        Returns
        -------
        Configuration
            The loaded configuration.
        """

        configs: List[Union[DictConfig, ListConfig]] = []

        # Set default values
        if defaults is not None:
            if not isinstance(defaults, list):
                defaults = [defaults]
            for d in defaults:
                if isinstance(d, str):
                    configs.append(OmegaConf.load(d))
                    continue
                configs.append(OmegaConf.create(d))

        # Load the user configuration
        if isinstance(path, dict):
            configs.append(OmegaConf.create(path))
        else:
            configs.append(OmegaConf.load(path))

        if not isinstance(overrides, list):
            overrides = [overrides]

        oc_config = OmegaConf.merge(*configs)

        for override in overrides:
            if isinstance(override, dict):
                oc_config = OmegaConf.merge(oc_config, OmegaConf.create(override))
            else:
                selected = oc_config
                key, value = override.split("=")
                keys = key.split(".")
                for key in keys[:-1]:
                    if key.isdigit() and isinstance(selected, ListConfig):
                        index = int(key)
                        if index < len(selected):
                            LOG.debug(f"key {key} is used as list index in list{selected}")
                            selected = selected[index]
                        elif index == len(selected):
                            LOG.debug(f"key {key} is used to append to list {selected}")
                            selected.append(OmegaConf.create())
                            selected = selected[index]
                        else:
                            raise IndexError(
                                f"Index {index} out of range for list {selected} of length {len(selected)}"
                            )
                    else:
                        selected = selected.setdefault(key, OmegaConf.create())
                selected[keys[-1]] = value

        resolved_config = OmegaConf.to_container(oc_config, resolve=True)

        # Validate the configuration
        config = cls.model_validate(resolved_config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config
