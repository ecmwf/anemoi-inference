# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections import defaultdict
from typing import Any

from anemoi.utils.humanize import plural
from earthkit.data.utils.dates import to_datetime

LOG = logging.getLogger(__name__)


def check_data(title: str, data: Any, variables: list[str], dates: list[datetime.datetime], checkpoint: Any) -> None:
    """Check if the data matches the expected number of fields based on variables and dates.

    Parameters
    ----------
    title : str
        The title for the data check.
    data : Any
        The data to be checked.
    variables : List[str]
        The list of variable names.
    dates : List[datetime.datetime]
        The list of dates.
    checkpoint : Any
        The checkpoint

    Raises
    ------
    ValueError
        If the data does not match the expected number of fields.
    """
    expected = len(variables) * len(dates)

    if len(data) != expected:

        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        console = Console()

        LOG.error("Data check failed for %s", title)

        nvars = plural(len(variables), "variable")
        ndates = plural(len(dates), "date")
        nfields = plural(expected, "field")
        msg = f"Expected ({nvars}) x ({ndates}) = {nfields}, got {len(data)}"
        LOG.error("%s", msg)

        table.add_column("Name", justify="left")

        dates = sorted(dates)
        variables = sorted(variables)

        for d in dates:
            table.add_column(d.isoformat(), justify="center")

        table.add_column("Categories")

        avail = defaultdict(set)
        duplicates = defaultdict(set)
        for field in data:
            name, date = field.metadata("name"), to_datetime(field.metadata("valid_datetime"))
            if date in avail[name]:
                duplicates[name].add(date)
                LOG.warning(
                    "Duplicate field for variable '%s' at date %s in %s",
                    name,
                    date.isoformat(),
                    title,
                )
            avail[name].add(date)

        variable_categories = checkpoint.variable_categories()
        for name in variables:
            row = [name]
            for d in dates:
                if d not in avail[name]:
                    row.append("❌")
                else:
                    if d in duplicates[name]:
                        row.append("⚠️")
                    else:
                        row.append("✅")

            if name in variable_categories:
                cats = ", ".join(sorted(variable_categories[name]))
                row.append(cats)
            else:
                row.append("N/A")

            table.add_row(*row)

        console.print()
        console.print(table)
        console.print()

        raise ValueError(msg)

    assert len(data) == len(variables) * len(dates)
