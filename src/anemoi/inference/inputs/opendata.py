# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data.utils.dates import to_datetime

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)

SOIL_MAPPING = {"stl1": "sot", "stl2": "sot", "stl3": "sot", "swvl1": "vsw", "swvl2": "vsw", "swvl3": "vsw"}


def postprocess_geopotential_height(fields: ekd.FieldList):
    """Change geopotential height to meters and rename to `z`."""
    r = ekd.FieldList()
    for field in fields:
        if field.metadata()["param"] == "gh":
            r += r.from_numpy(field.to_numpy() * 9.80665, field.metadata().override(paramId=shortname_to_paramid("z")))
        else:
            r += r.from_numpy(field.to_numpy(), field.metadata())
    return r


def _retrieve_soil(request: dict, soil_params: list[str]):
    """Retrieve soil params"""
    levels = list(set(int(s[-1]) for s in soil_params))
    request["param"] = list(SOIL_MAPPING[s] for s in soil_params)
    request["levelist"] = levels
    request.pop("levtype")

    soil_data = ekd.from_source("ecmwf-open-data", request)
    for field in soil_data:
        newname = {f"{v}{k[-1]}": k for k, v in SOIL_MAPPING.items()}[
            f"{field.metadata()['param']}{field.metadata()['level']}"
        ]
        field._metadata = field.metadata().override(paramId=shortname_to_paramid(newname))

    return soil_data


def regridding(fields: ekd.FieldList, grid: str, template):
    """Apply regridding to the field."""
    import earthkit.regrid as ekr
    import numpy as np

    r = ekd.FieldList()

    f_md = template.template(variable="tp", date=None).metadata()

    for f in fields:
        rolled_values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
        interpolated_values = ekr.interpolate(rolled_values, in_grid={"grid": (0.25, 0.25)}, out_grid={"grid": grid})

        # Set the metadata with the grid directly, not this ridiculous template, TODO Harrison Cook
        namespace_metadata = f.metadata().as_namespace("mars")
        namespace_metadata.update(f.metadata().as_namespace("time"))
        namespace_metadata["paramId"] = shortname_to_paramid(namespace_metadata.pop("param"))

        for k in ["typeOfLevel", "time", "date"]:
            namespace_metadata[k] = f.metadata()[k]
        for k in ["domain", "levtype", "step", "validityDate", "validityTime"]:
            namespace_metadata.pop(k)

        r += r.from_numpy(np.expand_dims(interpolated_values, 0), f_md.override(**namespace_metadata))
    return r


def retrieve(requests, grid, area, template, **kwargs):

    def _(r):
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    post_processing_functions = []

    result = ekd.from_source("empty")
    for r in requests:
        if "z" in r["param"] and r["levtype"] == "pl":
            r["param"] = ("gh", *(p for p in r.get("param") if p != "z"))
            post_processing_functions.append(postprocess_geopotential_height)

        # r.update(pproc)
        r.update(kwargs)

        if any(k in r["param"] for k in SOIL_MAPPING.keys()):
            requested_soil_variables = [k for k in SOIL_MAPPING.keys() if k in r["param"]]
            r["param"] = [p for p in r["param"] if p not in requested_soil_variables]
            result += regridding(_retrieve_soil(r.copy(), requested_soil_variables), grid, template)

        LOG.debug("%s", _(r))

        returned_data = ekd.from_source("ecmwf-open-data", r)
        for func in post_processing_functions:
            returned_data = func(returned_data)

        result += regridding(returned_data, grid, template)

    return result


@input_registry.register("opendata")
class OpenDataInput(GribInput):
    """Get input fields from ECMWF open-data"""

    def __init__(self, context, *, namer=None, **kwargs):
        super().__init__(context, namer=namer)

        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.kwargs = kwargs

    def create_input_state(self, *, date):
        if date is None:
            date = to_datetime(-1)
            LOG.warning("OpenDataInput: `date` parameter not provided, using yesterday's date: %s", date)

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

        kwargs = self.kwargs.copy()

        from .templates import TemplatesInput

        return retrieve(
            requests, self.checkpoint.grid, self.checkpoint.area, template=TemplatesInput(self.context), **kwargs
        )

    def template(self, variable, date, **kwargs):
        return self.retrieve([variable], [date])[0]

    def load_forcings(self, variables, dates):
        return self._load_forcings(self.retrieve(variables, dates), variables, dates)
