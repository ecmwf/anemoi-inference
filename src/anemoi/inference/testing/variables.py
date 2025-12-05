# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.transform.variables import Variable

tp = Variable.from_dict(
    "tp",
    {
        "mars": {
            "param": "tp",
            "levtype": "sfc",
        },
        "process": "accumulation",
        "period": [0, 6],
    },
)

z = Variable.from_dict(
    "z",
    {
        "mars": {
            "param": "z",
            "levtype": "sfc",
        }
    },
)

w_100 = Variable.from_dict(
    "w_100",
    {
        "mars": {
            "param": "w",
            "levtype": "pl",
            "levelist": 100,
        }
    },
)
