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
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from anemoi.inference.context import Context

from ..processor import Processor
from . import pre_processor_registry

LOG = logging.getLogger(__name__)


@pre_processor_registry.register("no_missing_values")
class NoMissingValues(Processor):
    """Replace NaNs with mean."""

    def __init__(self, context: Context, **kwargs: Any) -> None:
        """Initialize the NoMissingValues processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context)

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Process the fields to replace NaNs with the mean value.

        Parameters
        ----------
        fields : list
            List of fields to process.

        Returns
        -------
        list
            List of processed fields with NaNs replaced by the mean value.
        """
        result = []
        for field in tqdm.tqdm(fields):
            import numpy as np

            data = field.to_numpy()

            mean_value = np.nanmean(data)

            data = np.where(np.isnan(data), mean_value, data)
            result.append(new_field_from_numpy(data, template=field))

        return new_fieldlist_from_list(result)
