# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from functools import cached_property

from anemoi.utils.checkpoints import load_metadata
from earthkit.data.utils.dates import to_datetime

from .metadata import Metadata

LOG = logging.getLogger(__name__)


class Checkpoint:
    """Represents an inference checkpoint."""

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return self.path

    @cached_property
    def _metadata(self):
        return Metadata(load_metadata(self.path))

    ###########################################################################
    # Forwards used by the runner
    # We do not want to expose the metadata object directly
    # We do not use `getattr` to avoid exposing all methods and make debugging
    # easier
    ###########################################################################

    @property
    def frequency(self):
        return self._metadata.frequency

    @property
    def precision(self):
        return self._metadata.precision

    @property
    def number_of_grid_points(self):
        return self._metadata.number_of_grid_points

    @property
    def number_of_input_features(self):
        return self._metadata.number_of_input_features

    @property
    def variable_to_input_tensor_index(self):
        return self._metadata.variable_to_input_tensor_index

    @property
    def model_computed_variables(self):
        return self._metadata.model_computed_variables

    @property
    def typed_variables(self):
        return self._metadata.typed_variables

    @property
    def diagnostic_variables(self):
        return self._metadata.diagnostic_variables

    @property
    def prognostic_output_mask(self):
        return self._metadata.prognostic_output_mask

    @property
    def prognostic_input_mask(self):
        return self._metadata.prognostic_input_mask

    @property
    def output_tensor_index_to_variable(self):
        return self._metadata.output_tensor_index_to_variable

    @property
    def computed_time_dependent_forcings(self):
        return self._metadata.computed_time_dependent_forcings

    ###########################################################################

    @cached_property
    def lagged(self):
        """Return the list of timedelta for the lagged input fields."""
        result = list(range(0, self._metadata.multi_step_input))
        result = [-s * self._metadata.frequency for s in result]
        return sorted(result)

    ###########################################################################
    # Data retrieval
    ###########################################################################

    @property
    def grid(self):
        return self._metadata.grid

    @property
    def area(self):
        return self._metadata.area

    def mars_requests(self, dates, use_grib_paramid=False, **kwargs):
        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        result = []

        for r in self._metadata.mars_requests(use_grib_paramid=use_grib_paramid, **kwargs):
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
