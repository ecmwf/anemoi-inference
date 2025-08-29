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
from typing import TypeVar

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

    date: datetime | None = None
    """The starting date for the forecast. If not provided, the date will depend on the selected Input object. If a string, it is parsed by :func:`earthkit.data.utils.dates`."""

    @field_validator("date", mode="before")
    @classmethod
    def to_datetime(cls, date: str | int | datetime | None) -> datetime | None:
        if date is not None:
            return to_datetime(date)

    @classmethod
    def load(
        cls: type[T],
        path: str | dict[str, Any],
        overrides: list[str] | list[dict] | str | dict = [],
        defaults: str | list[str] | dict | None = None,
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

        configs: list[DictConfig | ListConfig] = []

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
                override_conf = OmegaConf.from_dotlist([override])
                # We can't directly merge reconstructed with the config because
                # omegaconf prefers parsing digits (like 0 in key.0) into dict keys
                # rather than lists.
                # Instead, we provide a reference config and we try to merge the override
                # into the reference and keep types provided by the reference.
                oc_config = OmegaConf.unsafe_merge(_merge_configs(oc_config, override_conf))

        resolved_config = OmegaConf.to_container(oc_config, resolve=True)

        config = cls.model_validate(resolved_config)

        # Set environment variables found in the configuration
        # as soon as possible
        for key, value in config.env.items():
            os.environ[key] = str(value)

        return config


def _merge_configs(ref_conf: Any, new_conf: Any) -> Any:
    """Recursively merges a new OmegaConf object into a reference OmegaConf object

    Parameters
    ----------
    ref_conf : Any
        reference OmegaConf object. Should be a DictConfig or ListConfig.
    new_conf : Any
        new OmegaConf object.

    Returns
    -------
    Any
        The merged OmegaConf config
    """
    if isinstance(new_conf, DictConfig) and len(new_conf):
        key, rest = next(iter(new_conf.items()))
        key = str(key)
    elif isinstance(new_conf, ListConfig) and len(new_conf):
        key, rest = 0, new_conf[0]
    else:
        return new_conf
    if isinstance(ref_conf, ListConfig):
        if isinstance(key, str) and not key.isdigit():
            raise ValueError(f"Expected int key, got {key}")
        index = int(key)
        if index < len(ref_conf):
            LOG.debug(f"key {key} is used as list key in list{ref_conf}")
            ref_conf[index] = _merge_configs(ref_conf[index], rest)
        elif index == len(ref_conf):
            LOG.debug(f"key {key} is used to append to list {ref_conf}")
            ref_conf.append(rest)
        else:
            raise IndexError(f"key {key} out of range for list {ref_conf} of length {len(ref_conf)}")
        return ref_conf
    elif isinstance(ref_conf, DictConfig) and key in ref_conf:
        ref_conf[key] = _merge_configs(ref_conf[key], rest)
        return ref_conf
    elif isinstance(ref_conf, DictConfig):
        ref_conf[key] = rest
        return ref_conf
    else:
        raise ValueError(f"ref is of unexpected type {type(ref_conf)}. Should be ListConfig or DictConfig")
