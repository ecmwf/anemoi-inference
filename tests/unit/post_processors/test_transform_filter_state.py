# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import numpy as np

from anemoi.inference.post_processors.earthkit_state import wrap_state


def test_wrapped_state_fields_expose_valid_datetime_and_grid() -> None:
    """Date- and grid-dependent anemoi-transform filters read these from the fields."""
    date = datetime.datetime(2016, 9, 22, 19)
    lat = np.array([0.0, 10.0, 20.0])
    lon = np.array([0.0, 90.0, 180.0])
    state = {"date": date, "latitudes": lat, "longitudes": lon, "fields": {"2t": np.zeros(3, dtype=np.float32)}}

    field = wrap_state(state, {"2t": SimpleNamespace(grib_keys={"param": "2t"})})[0]

    assert field.time.valid_datetime() == date
    lats, lons = field.geography.latlons()
    assert lats.tolist() == lat.tolist()
    assert lons.tolist() == lon.tolist()
