# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
import os
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

import yaml
from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Configuration(BaseModel):

    class Config:
        extra = "forbid"

    description: Optional[str] = None

    checkpoint: str | Dict[Literal["huggingface"], Dict[str, Any] | str]
    """A path to an Anemoi checkpoint file."""

    date: Union[str, int, datetime.datetime, None] = None
    """The starting date for the forecast. If not provided, the date will depend on the selected Input object. If a string, it is parsed by :func:`anemoi.utils.dates.as_datetime`.
    """

    lead_time: str | int | datetime.timedelta = "10d"
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    name: Optional[str] = None
    """Used by prepml."""

    verbosity: int = 0
    """The verbosity level of the runner. This can be 0 (default), 1, 2 or 3."""

    report_error: bool = False
    """If True, the runner list the training versions of the packages in case of error."""

    input: Union[str, Dict, None] = "test"
    output: Union[str, Dict, None] = "printer"

    forcings: Union[Dict[str, Dict], None] = None
    """Where to find the forcings."""

    device: str = "cuda"
    """The device on which the model should run. This can be "cpu", "cuda" or any other value supported by PyTorch."""

    precision: Optional[str] = None
    """The precision in which the model should run. If not provided, the model will use the precision used during training."""

    allow_nans: Optional[bool] = None
    """
    - If None (default), the model will check for NaNs in the input. If NaNs are found, the model issue a warning and `allow_nans` to True.
    - If False, the model will raise an exception if NaNs are found in the input and output.
    - If True, the model will allow NaNs in the input and output.
    """

    use_grib_paramid: bool = False
    """If True, the runner will use the grib parameter ID when generating MARS requests."""

    write_initial_state: bool = True
    """Wether to write the initial state to the output file. If the model is multi-step, only fields at the forecast reference date are
    written."""

    env: Dict[str, str | int] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured."""

    patch_metadata: dict = {}
    """A dictionary of metadata to patch the checkpoint metadata with. This is used to test new features or to work around
    issues with the checkpoint metadata."""

    development_hacks: dict = {}
    """A dictionary of development hacks to apply to the runner. This is used to test new features or to work around"""

    debugging_info: dict = {}
    """A dictionary to store debug information. This is ignored."""


def load_config(path, overrides, defaults=None, Configuration=Configuration):

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
    with open(path) as f:
        config.update(yaml.safe_load(f))

    # Apply overrides
    for override in overrides:
        path = config
        key, value = override.split("=")
        keys = key.split(".")
        for key in keys[:-1]:
            path = path.setdefault(key, {})
        path[keys[-1]] = value

    # Validate the configuration
    config = Configuration(**config)

    # Set environment variables found in the configuration
    # as soon as possible
    for key, value in config.env.items():
        os.environ[key] = str(value)

    return config
