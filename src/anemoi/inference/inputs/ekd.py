# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from collections import defaultdict

import numpy as np
from anemoi.utils.humanize import plural
from earthkit.data.indexing.fieldlist import FieldArray

from . import Input

LOG = logging.getLogger(__name__)


class EkdInput(Input):
    """
    Handles earthkit-data FieldList as input
    """

    def __init__(self, runner, *, namer=None):
        super().__init__(runner)
        self._namer = namer if namer is not None else self.checkpoint.default_namer()
        assert callable(self._namer), type(self._namer)

    def _create_input_state(
        self, input_fields, date=None, latitudes=None, longitudes=None, dtype=np.float32, flatten=True
    ):

        # Newer checkpoints may have latitudes and longitudes
        if latitudes is None:
            latitudes = self.checkpoint.latitudes
            if latitudes is not None:
                LOG.info(
                    "%s: using `latitudes` found in the checkpoint.",
                    self.__class__.__name__,
                )

        if longitudes is None:
            longitudes = self.checkpoint.longitudes
            if longitudes is not None:
                LOG.info(
                    "%s: using `longitudes` found in the checkpoint.git c",
                    self.__class__.__name__,
                )

        if date is None:
            date = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info(
                "%s: `date` not provided, using the most recent date: %s", self.__class__.__name__, date.isoformat()
            )

        dates = [date + h for h in self.checkpoint.lagged]
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        input_state = dict(reference_date=date, date=date, latitudes=latitudes, longitudes=longitudes, fields=dict())

        fields = input_state["fields"]

        input_fields = self._filter_and_sort(input_fields, dates)

        check = defaultdict(set)

        for field in input_fields:

            if input_state["latitudes"] is None:
                input_state["latitudes"], input_state["longitudes"] = field.grid_points()
                LOG.info(
                    "%s: using `latitudes` and `longitudes` from the first input field",
                    self.__class__.__name__,
                )

            name, valid_datetime = field.metadata("name"), field.metadata("valid_datetime")
            if name not in fields:
                fields[name] = np.full(
                    shape=(len(dates), self.checkpoint.number_of_grid_points),
                    fill_value=np.nan,
                    dtype=dtype,
                )

            date_idx = date_to_index[valid_datetime]

            try:
                fields[name][date_idx] = field.to_numpy(dtype=dtype, flatten=flatten)
            except ValueError:
                LOG.error("Error with field %s: expected shape=%s, got shape=%s", name, fields[name].shape, field.shape)
                LOG.error("dates %s", dates)
                LOG.error("number_of_grid_points %s", self.checkpoint.number_of_grid_points)
                raise

            if date_idx in check[name]:
                LOG.error("Duplicate dates for %s: %s", name, date_idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(check[name]))
                raise ValueError(f"Duplicate dates for {name}")

            check[name].add(date_idx)

        for name, idx in check.items():
            if len(idx) != len(dates):
                LOG.error("Missing dates for %s: %s", name, idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(idx))
                raise ValueError(f"Missing dates for {name}")

        # This is our chance to communicate output object
        # This is useful for GRIB that requires a template field
        # to be used as output
        self.set_private_attributes(input_state, input_fields)

        return input_state

    def _filter_and_sort(self, data, dates):
        typed_variables = self.checkpoint.typed_variables
        diagnostic_variables = self.checkpoint.diagnostic_variables

        def _name(field, _, original_metadata):
            return self._namer(field, original_metadata)

        data = FieldArray([f.copy(name=_name) for f in data])

        variable_from_input = [
            v.name for v in typed_variables.values() if v.is_from_input and v.name not in diagnostic_variables
        ]

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        data = data.sel(name=variable_from_input, valid_datetime=valid_datetime).order_by("name", "valid_datetime")

        expected = len(variable_from_input) * len(dates)

        if len(data) != expected:
            nvars = plural(len(variable_from_input), "variable")
            ndates = plural(len(dates), "date")
            nfields = plural(expected, "field")
            msg = f"Expected ({nvars}) x ({ndates}) = {nfields}, got {len(data)}"
            LOG.error("%s", msg)
            # TODO: print a report
            raise ValueError(msg)

        assert len(data) == len(variable_from_input) * len(dates)

        return data
