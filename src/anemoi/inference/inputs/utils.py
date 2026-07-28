# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.data.utils.dates import to_datetime

from anemoi.inference.types import Date


def convert_dates_to_base_and_step(dates: list[Date], base_date: Date | None = None) -> tuple[Date, list[int]]:
    """Convert a list of dates to base and step.

    Parameters
    ----------
    dates : list[Date]
        List of dates to convert.
    base_date : Date or None
        The base date to use. If None, the earliest date in the list is used.

    Returns
    -------
    tuple[Date, list[int]]
        The base date and the list of steps in hours.
    """
    if not dates:
        raise ValueError("The list of dates is empty.")
    datetimes = [to_datetime(date) for date in dates]

    base_date = to_datetime(base_date) if base_date else min(datetimes)
    steps = [int((dt - to_datetime(base_date)).total_seconds() // 3600) for dt in datetimes]

    return base_date, steps
