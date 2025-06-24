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
from typing import List
from typing import Optional
from typing import Tuple

from anemoi.inference.config import Configuration

LOG = logging.getLogger(__name__)


class CoupleConfiguration(Configuration):
    """Configuration class for the couple runner."""

    description: Optional[str] = None

    lead_time: Optional[Tuple[str, int, datetime.timedelta]] = None
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    name: Optional[str] = None
    """Used by prepml."""

    transport: str
    couplings: List[Dict[str, List[str]]]
    tasks: Dict[str, Any]

    env: Dict[str, Any] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured.
    """
