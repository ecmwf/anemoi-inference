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
    """The starting date for the forecast."""

    device: str = "cuda"
    lead_time: str | int = "10d"
    precision: str | None = None
    allow_nans: bool = False
    icon_grid: str | None = None
    input: str | None = None
    output: str | None = None
    write_initial_state: bool = True
    use_grib_paramid: bool = False
    dataset: bool = False
    env: Dict[str, str] = {}
