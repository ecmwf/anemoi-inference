# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import earthkit.data as ekd
import tqdm
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data.sources.array_list import from_array

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest

from ..processor import Processor
from . import pre_processor_registry

LOG = logging.getLogger(__name__)

try:
    from earthkit.meteo.constants import g as G
except ImportError:
    LOG.warning("Could not import g from earthkit.meteo.constants. Using default value.")
    G = 9.80665


@pre_processor_registry.register("geopotential_height")
class GeopotentialHeight(Processor):
    """Change geopotential height to meters and rename to `z`."""

    def __init__(self, context: Context, **kwargs: Any) -> None:
        """Initialize the GeopotentialHeight processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context)

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Process the fields to replace geopotential with z.

        Parameters
        ----------
        fields : list
            List of fields to process.

        Returns
        -------
        ekd.FieldList
            Processed fields with gh replaced by z.
        """
        result = ekd.FieldList()
        for field in tqdm.tqdm(fields):
            if field.metadata()["param"] == "gh":
                result += from_array(field.to_numpy() * G, field.metadata().override(paramId=shortname_to_paramid("z")))
            else:
                result += from_array(field.to_numpy(), field.metadata())

        return result

    def patch_data_request(self, data_request: DataRequest) -> DataRequest:
        """Patch the data request to call for geopotential height instead of z.

        Parameters
        ----------
        data_request : DataRequest
            The data request to be patched.

        Returns
        -------
        DataRequest
            The patched data request.
        """
        if "z" in data_request["param"] and data_request["levtype"] == "pl":
            data_request["param"] = ("gh", *(p for p in data_request.get("param") if p != "z"))
        return data_request
