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
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:

    from .checkpoint import Checkpoint

LOG = logging.getLogger(__name__)


class Context(ABC):
    """Represents the context of the inference."""

    allow_nans = None  # can be True of False
    use_grib_paramid = False
    verbosity = 0
    development_hacks: dict[str, Any] = {}  # For testing purposes, don't use in production

    # Some runners will set these values, which can be queried by Output objects,
    # but may remain as None

    reference_date = None
    time_step = None
    lead_time = None
    output_frequency: int | None = None
    write_initial_state: bool = True
    writers_per_device: int = 0

    ##################################################################

    @property
    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Returns the checkpoint used for the inference."""
        pass
