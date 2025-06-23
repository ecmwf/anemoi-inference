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

        # unsafe merge should be fine as we don't re-use the original configs
        oc_config = OmegaConf.unsafe_merge(*configs)

        for override in overrides:
            if isinstance(override, dict):
                oc_config = OmegaConf.unsafe_merge(oc_config, OmegaConf.create(override))
            else:
                # use from_dotlist to use OmegaConf split
                # which allows for "param.val" or "param[val]".
                reconstructed = OmegaConf.from_dotlist([override])
                oc_config = OmegaConf.unsafe_merge(_merge_dicts(oc_config, reconstructed))

        resolved_config = OmegaConf.to_container(oc_config, resolve=True)

        config = cls.model_validate(resolved_config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config


def _merge_dicts(ref: Any, new: Any) -> Any:
    """Recursively merge a new OmegaConf object into a reference OmegaConf object

    Parameters
    ----------
    ref : Any
        reference OmegaConf object. Should be a DictConfig or ListConfig.
    new : Any
        new OmegaConf object.

    Returns
    -------
    Any
        The merged OmegaConf config
    """
    if isinstance(new, DictConfig):
        key, rest = next(iter(new.items()))
        key = str(key)
    elif isinstance(new, ListConfig):
        key, rest = 0, new[0]
    else:
        return new
    if isinstance(ref, ListConfig):
        if isinstance(key, str) and not key.isdigit():
            raise ValueError(f"Expected int key, got {key}")
        index = int(key)
        if index < len(ref):
            LOG.debug(f"key {key} is used as list key in list{ref}")
            ref[index] = _merge_dicts(ref[index], rest)
        elif index == len(ref):
            LOG.debug(f"key {key} is used to append to list {ref}")
            ref.append(rest)
        else:
            raise IndexError(f"key {key} out of range for list {ref} of length {len(ref)}")
        return ref
    elif isinstance(ref, DictConfig):
        ref[key] = ref.setdefault(key, OmegaConf.create())
        ref[key] = _merge_dicts(ref[key], rest)
        return ref
    else:
        raise ValueError(f"ref is of unexpected type {type(ref)}. Should be ListConfig or DictConfig")
