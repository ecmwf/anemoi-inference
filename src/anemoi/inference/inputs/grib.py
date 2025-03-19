# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc
import datetime
import logging
from typing import Any
from typing import Dict

import earthkit.data as ekd
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.readers import Reader

from .ekd import EarthKitInput

LOG = logging.getLogger(__name__)


class GribInput(EarthKitInput):
    """Handles grib files."""

    def _raw_state_fieldlist(self, dates: list[datetime.datetime], variables: list[str]) -> ekd.FieldList:
        """Load the raw state fieldlist for the given dates and variables."""

        # NOTE: this is deferred because it's only passed as a callable
        def _name(field: Any, _: Any, original_metadata: Dict[str, Any]) -> str:
            return self._namer(field, original_metadata)

        data = ekd.from_source(self.path)
        data = FieldArray([f.clone(name=_name) for f in data])

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        # it's actually executed here
        data = data.sel(name=variables, valid_datetime=valid_datetime).order_by(
            name=variables, valid_datetime="ascending"
        )

        return data

    @abc.abstractmethod
    def _earthkit_reader(self, *args, **kwargs) -> Reader:
        """Return the earthkit reader for the input."""
        pass
