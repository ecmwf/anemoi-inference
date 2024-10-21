# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
from functools import cached_property

from anemoi.utils.checkpoints import has_metadata as has_metadata
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.provenance import gather_provenance_info as gather_provenance_info
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.metadata import Metadata

LOG = logging.getLogger(__name__)


class Checkpoint:
    """Represents an inference checkpoint. Provides dot-notation access to the checkpoint's metadata."""

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return self.path

    @cached_property
    def metadata(self):
        return Metadata(load_metadata(self.path))

    @property
    def multi_step(self):
        return self.metadata.multi_step

    @property
    def hour_steps(self):
        return self.metadata.hour_steps

    @property
    def retrieve_request(self, *args, **kwargs):
        return self.metadata.retrieve_request(*args, **kwargs)

    @property
    def grid(self):
        return self.metadata.grid

    @property
    def area(self):
        return self.metadata.rounded_area(self.metadata.area)

    @property
    def precision(self):
        return self.metadata.precision

    # @property
    # def select(self):
    #     return self.metadata.select

    # @property
    # def order_by(self):
    #     return self.metadata.order_by

    @property
    def number_of_grid_points(self):
        return self.metadata.number_of_grid_points

    def filter_and_sort(self, data, dates):
        return self.metadata.filter_and_sort(data, dates)

    def mars_requests(self, dates, use_grib_paramid=False, **kwargs):
        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        result = []

        for r in self.metadata.retrieve_request(use_grib_paramid=use_grib_paramid):
            for date in dates:

                r = r.copy()

                base = date
                step = str(r.get("step", 0)).split("-")[-1]
                step = int(step)
                base = base - datetime.timedelta(hours=step)

                r["date"] = base.strftime("%Y-%m-%d")
                r["time"] = base.strftime("%H%M")

                r.update(kwargs)

                result.append(r)

        return result

    def summary(self):
        pass
