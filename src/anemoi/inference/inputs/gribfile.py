# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import List
from typing import Optional

import earthkit.data as ekd

from ..decorators import main_argument
from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("grib")
@main_argument("path")
class GribFileInput(GribInput):
    """Handles grib files."""

    trace_name = "grib file"

    def __init__(self, context: Any, path: str, *, namer: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(context, namer=namer, **kwargs)
        self.path = path

    def create_input_state(self, *, date: Optional[Any]) -> Any:
        return self._create_state(ekd.from_source("file", self.path), variables=None, date=date)

    def load_forcings_state(self, *, variables: List[str], dates: List[Any], current_state: Any) -> Any:
        return self._load_forcings_state(
            ekd.from_source("file", self.path), variables=variables, dates=dates, current_state=current_state
        )
