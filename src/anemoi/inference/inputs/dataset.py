# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
from earthkit.data.utils.dates import to_datetime

from . import Input

LOG = logging.getLogger(__name__)


class DatasetInput(Input):
    """
    Handles anemoi dataset as input
    """

    def __init__(self, context, /, *args, keep_paths=True, **kwargs):

        from anemoi.datasets import open_dataset

        super().__init__(context)

        # TODO: pass `keep_paths` as an argument

        keep_paths = False

        if not args and not kwargs:
            args, kwargs = self.checkpoint.open_dataset_args_kwargs(keep_paths=keep_paths)

            # TODO: remove start/end from the arguments

            LOG.warning("No arguments provided to open_dataset, using the default arguments:")
            LOG.warning("open_dataset(*%s, **%s)", args, kwargs)

        self.ds = open_dataset(*args, **kwargs)

    def create_input_state(self, *, date=None):
        if date is None:
            raise ValueError("`date` must be provided")

        date = to_datetime(date)

        input_state = dict(
            date=date,
            latitudes=self.ds.latitudes,
            longitudes=self.ds.longitudes,
            fields=dict(),
        )

        fields = input_state["fields"]

        dataset_dates = self.ds.dates
        date = np.datetime64(date)

        # TODO: use the fact that the dates are sorted

        idx = []
        for d in [date + np.timedelta64(h) for h in self.checkpoint.lagged]:
            (i,) = np.where(dataset_dates == d)
            if len(i) == 0:
<<<<<<< Updated upstream
                raise ValueError(
                    f"Date {d} not found in the dataset, available dates: {dataset_dates[0]}...{dataset_dates[-1]} by {self.ds.frequency}"
                )
=======
                raise ValueError(f"Date {d} not found in the dataset, available dates: {dataset_dates[0]}...{dataset_dates[-1]} by {self.ds.frequency}")
>>>>>>> Stashed changes
            assert len(i) == 1, f"Multiple dates found for {d}"
            idx.append(int(i[0]))

        if len(idx) == 1:
            s = slice(idx[0], idx[0] + 1)
        else:
            diff = idx[1] - idx[0]
            if not all(i == diff for i in np.diff(idx)):
                raise ValueError("Dates do not have the same frequency")
            s = slice(idx[0], idx[-1] + 1, diff)

        data = self.ds[s]

        if data.shape[2] != 1:
            raise ValueError(f"Ensemble data not supported, got {data.shape[2]} members")

        requested_variables = set(self.input_variables())
        for i, variable in enumerate(self.ds.variables):
            if variable not in requested_variables:
                continue
            # Squeeze the data to remove the ensemble dimension
            fields[variable] = np.squeeze(data[:, i], axis=1)

        return input_state
