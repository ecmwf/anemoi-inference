# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from collections import defaultdict
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
    def accumulations(self):
        return self._metadata.accumulations

    def default_namer(self, *args, **kwargs):
        """
        Return a callable that can be used to name fields.
        In that case, return the namer that was used to create the
        training dataset.
        """
        return self._metadata.default_namer(*args, **kwargs)

    def report_error(self):
        self._metadata.report_error()

    def open_dataset_args_kwargs(self):
        return self._metadata.open_dataset_args_kwargs()

    def dynamic_forcings_sources(self, runner):
        return self._metadata.dynamic_forcings_sources(runner)

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

    def mars_requests(self, dates, use_grib_paramid=False, variables=all, **kwargs):
        from earthkit.data.utils.availability import Availability

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        assert dates, "No dates provided"

        result = []

        DEFAULT_KEYS = ("class", "expver", "type", "stream", "levtype")
        DEFAULT_KEYS_AND_TIME = ("class", "expver", "type", "stream", "levtype", "time")

        # The split oper/scda is a bit special
        KEYS = {("oper", "fc"): DEFAULT_KEYS_AND_TIME, ("scda", "fc"): DEFAULT_KEYS_AND_TIME}

        requests = defaultdict(list)

        for r in self._metadata.mars_requests(use_grib_paramid=use_grib_paramid, variables=variables):
            for date in dates:

                r = r.copy()

                base = date
                step = str(r.get("step", 0)).split("-")[-1]
                step = int(step)
                base = base - datetime.timedelta(hours=step)

                r["date"] = base.strftime("%Y-%m-%d")
                r["time"] = base.strftime("%H%M")

                r.update(kwargs)  # We do it here so that the Availability can use that information

                keys = KEYS.get((r.get("stream"), r.get("type")), DEFAULT_KEYS)
                key = tuple(r.get(k) for k in keys)

                # Special case because of oper/scda

                requests[key].append(r)

        result = []
        for reqs in requests.values():

            compressed = Availability(reqs)
            for r in compressed.iterate():
                for k, v in r.items():
                    if isinstance(v, (list, tuple)) and len(v) == 1:
                        r[k] = v[0]
                if r:
                    result.append(r)

        return result
