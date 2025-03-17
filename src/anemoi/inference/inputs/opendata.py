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
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import State

from ..grib.templates import TemplateProvider
from ..grib.templates import create_template_provider
from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)

SOIL_MAPPING = {"stl1": "sot", "stl2": "sot", "stl3": "sot", "swvl1": "vsw", "swvl2": "vsw", "swvl3": "vsw"}


def _retrieve_soil(request: dict, soil_params: list[str]) -> ekd.FieldList:
    """Retrieve soil data.

    Map the soil parameters to the correct ECMWF parameter IDs and levels.

    Parameters
    ----------
    request : dict
        Request for the soil data.
    soil_params : list[str]
        Parameters to be retrieved.

    Returns
    -------
    ekd.FieldList
        Soil data.
    """
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


def regridding(
    fields: ekd.FieldList, grid: Optional[Union[str, List[float]]], area: Optional[List[float]], template: ekd.Field
) -> ekd.FieldList:
    """Apply regridding to the field.

    Parameters
    ----------
    fields : ekd.FieldList
        Fields to be regridded.
    grid : Optional[Union[str, List[float]]]
        Grid for the regridding.
    area : Optional[List[float]]
        Area for the regridding.
    template : ekd.Field
        Template for the regridding.

    Returns
    -------
    ekd.FieldList
        Regridded fields.
    """
    import earthkit.regrid as ekr
    import numpy as np

    r = ekd.FieldList()

    _ = area

    f_md = template.metadata()

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


def retrieve(
    requests: List[Dict[str, Any]],
    grid: Optional[Union[str, List[float]]],
    area: Optional[List[float]],
    template_provider: TemplateProvider,
    patch: Optional[Any] = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Retrieve data from ECMWF Opendata.

    Parameters
    ----------
    requests : List[Dict[str, Any]]
        The list of requests to be retrieved.
    grid : Optional[Union[str, List[float]]]
        The grid for the retrieval.
    area : Optional[List[float]]
        The area for the retrieval.
    template_provider : TemplateProvider
        Template provider for the regridding.
    patch : Optional[Any], optional
        Optional patch for the request, by default None.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The retrieved data.
    """

    def _(r: DataRequest):
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    result = ekd.from_source("empty")
    for r in requests:
        r.update(kwargs)

        if patch:
            r = patch(r)

        template = template_provider.template(None, {"grid": grid, "levtype": r["levtype"]})

        if any(k in r["param"] for k in SOIL_MAPPING.keys()):
            requested_soil_variables = [k for k in SOIL_MAPPING.keys() if k in r["param"]]
            r["param"] = [p for p in r["param"] if p not in requested_soil_variables]
            result += regridding(_retrieve_soil(r.copy(), requested_soil_variables), grid, area, template)

        LOG.debug("%s", _(r))
        result += regridding(ekd.from_source("ecmwf-open-data", r), grid, area, template)

    return result


@input_registry.register("opendata")
class OpenDataInput(GribInput):
    """Get input fields from ECMWF open-data.

    This will require the use of the `geopotential_height` pre-processor to
    rename any requests for `z` to `gh` and change the geopotential height to meters.
    ```yaml
    pre_processors:
      - geopotential_height
    ```
    """

    trace_name = "opendata"

    def __init__(
        self, context: Context, *, namer: Optional[Any] = None, templates: Optional[Union[str, dict]] = None, **kwargs
    ):
        """Initialize the OpenDataInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Any]
            Optional namer for the input.
        templates : Optional[Union[str, dict]], optional
            Optional specification for template provider, by default None
        """
        super().__init__(context, namer=namer)

        self.variables = self.checkpoint.variables_from_input(include_forcings=False)
        self.kwargs = kwargs

        self.template_provider = create_template_provider(self, templates or "builtin")

    def create_input_state(self, *, date: Optional[Date]) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.

        Returns
        -------
        State
            The created input state.
        """
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

    def retrieve(self, variables: List[str], dates: List[Date]) -> Any:
        """Retrieve data for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables to retrieve.
        dates : List[Any]
            The list of dates for which to retrieve the data.

        Returns
        -------
        Any
            The retrieved data.
        """

        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=self.context.use_grib_paramid,
        )

        if not requests:
            raise ValueError("No requests for %s (%s)" % (variables, dates))

        kwargs = self.kwargs.copy()

        return retrieve(
            requests,
            self.checkpoint.grid,
            self.checkpoint.area,
            template_provider=self.template_provider,
            patch=self.context.patch_data_request,
            **kwargs,
        )

    def load_forcings_state(self, *, variables: List[str], dates: List[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables for which to load the forcings state.
        dates : List[Date]
            The list of dates for which to load the forcings state.
        current_state : State
            The current state to be updated with the loaded forcings state.

        Returns
        -------
        Any
            The loaded forcings state.
        """
        return self._load_forcings_state(
            self.retrieve(variables, dates), variables=variables, dates=dates, current_state=current_state
        )
