# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import earthkit.data as ekd
import numpy as np

from anemoi.inference.inputs.ekd import _scalar_metadata_value
from anemoi.inference.outputs.netcdf import NetCDFOutput


def _context():
    checkpoint = SimpleNamespace(typed_variables={})
    return SimpleNamespace(
        checkpoint=checkpoint,
        reference_date=None,
        output_frequency=None,
        write_initial_state=True,
        typed_variables={},
    )


def test_ekd_compatible_output_writes_reference_time_attrs(tmp_path):
    output = NetCDFOutput(
        _context(),
        path=tmp_path / "netcdf" / "{baseDateTime}_{step:03}.nc",
        split_output=True,
        ekd_compatible=True,
    )

    state = {
        "latitudes": np.array([10.0, 20.0], dtype=np.float32),
        "longitudes": np.array([30.0, 40.0], dtype=np.float32),
        "fields": {
            "2t": np.array([280.0, 281.0], dtype=np.float32),
            "msl": np.array([100000.0, 100100.0], dtype=np.float32),
        },
        "date": datetime(2020, 1, 1, 6, 0),
        "step": timedelta(hours=6),
    }

    output.write_step(state)

    fields = ekd.from_source("file", str(tmp_path / "netcdf" / "202001010000_006.nc"))

    assert len(fields) == 2
    for field in fields:
        assert field.metadata("dataDate") == 20200101
        assert field.metadata("dataTime") == 0

    selected = fields.sel(
        dataDate=lambda value: _scalar_metadata_value(value) == 20200101,
        dataTime=lambda value: _scalar_metadata_value(value) == 0,
    )
    assert len(selected) == 2
