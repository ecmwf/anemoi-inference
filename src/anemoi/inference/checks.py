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
from typing import Any
from typing import List

from anemoi.utils.humanize import plural

LOG = logging.getLogger(__name__)


def check_data(title: str, data: Any, variables: List[str], dates: List[datetime.datetime]) -> None:
    expected = len(variables) * len(dates)

    if len(data) != expected:

        from anemoi.utils.text import table

        LOG.error("Data check failed for %s", title)

        nvars = plural(len(variables), "variable")
        ndates = plural(len(dates), "date")
        nfields = plural(expected, "field")
        msg = f"Expected ({nvars}) x ({ndates}) = {nfields}, got {len(data)}"
        LOG.error("%s", msg)

        cols = {}
        rows = {}
        t = []
        for i, d in enumerate(sorted(dates)):
            cols[d.isoformat()] = i + 1

        for i in range(len(variables)):
            name = variables[i]
            while len(t) <= i:
                t.append([name] + (["❌"] * len(cols)))

            t[i][0] = name
            rows[name] = i

        for field in data:
            name, date = field.metadata("name"), field.metadata("valid_datetime")
            if t[rows[name]][cols[date]] == "❌":
                t[rows[name]][cols[date]] = ""
            t[rows[name]][cols[date]] += "✅"

        print(table(t, ["name"] + [_.isoformat() for _ in sorted(dates)], "<||"))

        raise ValueError(msg)

    assert len(data) == len(variables) * len(dates)
