# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import datetime
import logging
from typing import Dict

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Configuration(BaseModel):

    checkpoint: str  # = "???"
    """A path an Anemoi checkpoint file."""

    date: str | int | datetime.datetime | None = None
    """The starting date for the forecast.

    If not provided, the date will depend on the selected Input object ."""

    device: str = "cuda"
    lead_time: str | int | datetime.timedelta = "10d"
    precision: str | None = None
    """The precision in which the model should run. If not provided, the model will use the precision used during training."""
    allow_nans: bool | None = None
    """"""
    icon_grid: str | None = None
    input: str | None = None
    output: str | None = None
    write_initial_state: bool = True
    use_grib_paramid: bool = False
    dataset: bool = False
    env: Dict[str, str] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured."""
