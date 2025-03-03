# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

# "2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitudes", "insolation"
#   0,   1,     2,     3,     4,    5,    6,      7,        8,            9


def mock_load_metadata(path: str, *, supporting_arrays=True, name: Any = None) -> dict:
    metadata = {
        "config": {
            "training": {"multistep_input": 2},
            "data": {"timestep": "6h", "forcing": ["lsm", "z"], "diagnostic": ["tcc", "tp"]},
        },
        "data_indices": {
            "data": {"input": {"full": [0, 1, 2, 3, 4, 5, 8, 9], "forcing": [4, 5, 8, 9], "diagnostic": [6, 7]}},
            "model": {"input": {"full": [0, 1, 2, 3, 4, 5, 6, 7], "forcing": [4, 5, 6, 7]}},
        },
        "dataset": {
            "variables": ["2t", "10u", "10v", "msl", "lsm", "z", "tcc", "tp", "cos_latitude", "insolation"],
            "shape": [365, 10, 1, 40320],
            "variables_metadata": {
                "2t": {},
                "10u": {},
                "10v": {},
                "msl": {},
                "lsm": {"constant_in_time": True},
                "z": {"constant_in_time": True},
                "tcc": {},
                "tp": {},
                "cos_latitude": {"computed_forcing": True, "constant_in_time": True},
                "insolation": {"computed_forcing": True, "constant_in_time": False},
            },
        },
    }

    arrays = {}

    if supporting_arrays:
        return metadata, arrays
    return metadata


def mock_torch_load(path: str, *, map_location=None):
    return None
