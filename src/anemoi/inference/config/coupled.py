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
from typing import Dict

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class CoupledConfiguration(BaseModel):

    class Config:
        extra = "forbid"

    description: str | None = None

    transport: str
    couplings: list[dict[str, list[str]]]
    tasks: dict[str, dict[str, dict[str, str]]]

    env: Dict[str, str | int] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured."""
