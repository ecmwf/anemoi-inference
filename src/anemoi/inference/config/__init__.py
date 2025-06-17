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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import ConfigDict

LOG = logging.getLogger(__name__)

T = TypeVar("T", bound="Configuration")


class Configuration(BaseModel):
    """Configuration class."""

    model_config = ConfigDict(extra="forbid")

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

        # Apply overrides
        if not isinstance(overrides, list):
            overrides = [overrides]

        for override in overrides:
            if isinstance(override, dict):
                configs.append(OmegaConf.create(override))
            else:
                configs.append(OmegaConf.from_dotlist([override]))

        resolved_config = OmegaConf.to_container(OmegaConf.merge(*configs), resolve=True)

        # Validate the configuration
        config = cls.model_validate(resolved_config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config
