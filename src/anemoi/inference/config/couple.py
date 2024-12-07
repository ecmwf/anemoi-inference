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
from typing import Any
from typing import Dict

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class CoupleConfiguration(BaseModel):

    class Config:
        extra = "forbid"

    description: str | None = None

    date: str | int | datetime.datetime | None = None
    """The starting date for the forecast. If not provided, the date will depend on the selected Input object. If a string, it is parsed by :func:`anemoi.utils.dates.as_datetime`.
    """

    lead_time: str | int | datetime.timedelta = None
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    name: str | None = None
    """Used by prepml."""

    transport: str
    couplings: list[dict[str, list[str]]]
    tasks: dict[str, dict[str, dict[str, Any]]]

    env: Dict[str, str | int] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured."""
