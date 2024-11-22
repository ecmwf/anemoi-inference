# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
from collections import defaultdict

import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.utils.dates import to_datetime

from ..checks import check_data
from ..input import Input

LOG = logging.getLogger(__name__)


class NoMask:
    """No mask to apply"""

    def apply(self, field):
        return field


class ApplyMask:
    """Apply a mask to a field"""

    def __init__(self, mask):
        self.mask = mask

    def apply(self, field):
        return field[self.mask]


class RulesNamer:
    """A namer that uses rules to generate names"""

    def __init__(self, rules, default_namer):
        self.rules = rules
        self.default_namer = default_namer

    def __call__(self, field, original_metadata):
        for rule in self.rules:
            assert len(rule) == 2, rule
            ok = True
            for k, v in rule[0].items():
                if original_metadata.get(k) != v:
                    ok = False
            if ok:
                return self.substitute(rule[1], field, original_metadata)

        return self.default_namer(field, original_metadata)

    def substitute(self, template, field, original_metadata):
        matches = re.findall(r"\{(.+?)\}", template)
        matches = {m: original_metadata.get(m) for m in matches}
        return template.format(**matches)


class EkdInput(Input):
    """Handles earthkit-data FieldList as input"""

    def __init__(self, context, *, namer=None):
        super().__init__(context)

        if isinstance(namer, dict):
            # TODO: a factory for namers
            assert "rules" in namer, namer
            assert len(namer) == 1, namer
            namer = RulesNamer(namer["rules"], self.checkpoint.default_namer())

        self._namer = namer if namer is not None else self.checkpoint.default_namer()
        assert callable(self._namer), type(self._namer)

    def _create_input_state(
        self,
        input_fields,
        *,
        variables=None,
        date=None,
        latitudes=None,
        longitudes=None,
        dtype=np.float32,
        flatten=True,
    ):

        if variables is None:
            variables = self.checkpoint.variables_from_input(include_forcings=True)

        if len(input_fields) == 0:
            raise ValueError("No input fields provided")

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
                    "%s: using `longitudes` found in the checkpoint.git.",
                    self.__class__.__name__,
                )

        if date is None:
            date = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info(
                "%s: `date` not provided, using the most recent date: %s", self.__class__.__name__, date.isoformat()
            )

        date = to_datetime(date)
        dates = [date + h for h in self.checkpoint.lagged]
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        input_state = dict(date=date, latitudes=latitudes, longitudes=longitudes, fields=dict())

        fields = input_state["fields"]

        input_fields = self._filter_and_sort(input_fields, variables=variables, dates=dates, title="Create input state")
        mask = self.checkpoint.grid_points_mask
        mask = ApplyMask(mask) if mask is not None else NoMask()

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
                fields[name][date_idx] = mask.apply(field.to_numpy(dtype=dtype, flatten=flatten))
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

    def _filter_and_sort(self, data, *, variables, dates, title):

        def _name(field, _, original_metadata):
            return self._namer(field, original_metadata)

        data = FieldArray([f.clone(name=_name) for f in data])

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        data = data.sel(name=variables, valid_datetime=valid_datetime).order_by(
            name=variables, valid_datetime="ascending"
        )

        check_data(title, data, variables, dates)

        return data

    def _find_variable(self, data, name, **kwargs):
        def _name(field, _, original_metadata):
            return self._namer(field, original_metadata)

        data = FieldArray([f.clone(name=_name) for f in data])
        return data.sel(name=name, **kwargs)

    def _load_forcings(self, fields, variables, dates):
        data = self._filter_and_sort(fields, variables=variables, dates=dates, title="Load forcings")
        return data.to_numpy(dtype=np.float32, flatten=True).reshape(len(variables), len(dates), -1)
