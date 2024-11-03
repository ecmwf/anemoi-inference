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
from typing import Dict

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Configuration(BaseModel):

    checkpoint: str
    """A path an Anemoi checkpoint file."""

    runner: str = "default"
    """Select the Runner."""

    date: str | int | datetime.datetime | None = None
    """The starting date for the forecast. If not provided, the date will depend on the selected Input object. If a string, it is parsed by :func:`anemoi.utils.dates.as_datetime`.
    """

    lead_time: str | int | datetime.timedelta = "10d"
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    verbosity: int = 0
    """The verbosity level of the runner. This can be 0 (default), 1, 2 or 3."""

    report_error: bool = False
    """If True, the runner list the training versions of the packages in case of error."""

    input: str | Dict | None = "mars"
    output: str | Dict | None = "printer"

    forcings: Dict[str, Dict] | None = None
    """Where to find the forcings."""

    device: str = "cuda"
    """The device on which the model should run. This can be "cpu", "cuda" or any other value supported by PyTorch."""

    precision: str | None = None
    """The precision in which the model should run. If not provided, the model will use the precision used during training."""

    allow_nans: bool | None = None
    """
    - If None (default), the model will check for NaNs in the input. If NaNs are found, the model issue a warning and `allow_nans` to True.
    - If False, the model will raise an exception if NaNs are found in the input and output.
    - If True, the model will allow NaNs in the input and output.
    """

    write_initial_state: bool = True
    """Wether to write the initial state to the output file. If the model is multi-step, only fields at the forecast reference date are
    written."""

    env: Dict[str, str | int] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured."""
