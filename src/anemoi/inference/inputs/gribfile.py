# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import input_registry
from .ekd import FieldlistInput
from .grib import GribInput


@input_registry.register("grib")
class GribFileInput(FieldlistInput, GribInput):
    """Handles grib files."""

    trace_name = "grib file"
    patterns = ("*.grib", "*.grb", "*.grb2", "*.grib2")
