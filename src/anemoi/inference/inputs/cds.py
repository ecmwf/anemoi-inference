# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from earthkit.data.utils.dates import to_datetime

from . import input_registry
from .grib import GribInput
from .mars import postproc

LOG = logging.getLogger(__name__)


def retrieve(requests, grid, area, dataset, **kwargs):
    import earthkit.data as ekd

    def _(r):
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    pproc = postproc(grid, area)

    result = ekd.from_source("empty")
    for r in requests:
        if isinstance(dataset, str):
            d = dataset
        elif isinstance(dataset, dict):
            # Get dataset from intersection of keys between request and dataset dict
            search_dataset = dataset.copy()
            while isinstance(search_dataset, dict):
                keys = set(r.keys()).intersection(set(search_dataset.keys()))
                if len(keys) == 0:
                    raise KeyError(
                        f"While searching for dataset, could not find any valid key in dictionary: {r.keys()}, {search_dataset}"
                    )
                key = list(keys)[0]
                if r[key] not in search_dataset[key]:
                    if "*" in search_dataset[key]:
                        search_dataset = search_dataset[key]["*"]
                        continue

                    raise KeyError(
                        f"Dataset dictionary does not contain key {r[key]!r} in {key!r}: {dict(search_dataset[key])}."
                    )
                search_dataset = search_dataset[key][r[key]]

            d = search_dataset

        r.update(pproc)
        r.update(kwargs)

        LOG.debug("%s", _(r))
        result += ekd.from_source("cds", d, r)

    return result


@input_registry.register("cds")
class CDSInput(GribInput):
    """Get input fields from CDS"""

    def __init__(self, context, *, dataset, namer=None, **kwargs):
        super().__init__(context, namer=namer)

        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.dataset = dataset
        self.kwargs = kwargs

    def create_input_state(self, *, date):
        if date is None:
            date = to_datetime(-1)
            LOG.warning("CDSInput: `date` parameter not provided, using yesterday's date: %s", date)

        date = to_datetime(date)

        return self._create_input_state(
            self.retrieve(
                self.variables,
                [date + h for h in self.checkpoint.lagged],
            ),
            variables=self.variables,
            date=date,
        )

    def retrieve(self, variables, dates):

        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
        )

        if not requests:
            raise ValueError("No requests for %s (%s)" % (variables, dates))

        return retrieve(
            requests, self.checkpoint.grid, self.checkpoint.area, dataset=self.dataset, expver="0001", **self.kwargs
        )

    def template(self, variable, date, **kwargs):
        return self.retrieve([variable], [date])[0]

    def load_forcings(self, variables, dates):
        return self._load_forcings(self.retrieve(variables, dates), variables, dates)
