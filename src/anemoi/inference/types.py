# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any
from typing import Dict

"""A collection of types used in the inference module. Some of these type could be moved to anemoi.utils.types or anemoi.transform.types."""

DataRequest = Dict[str, Any]
"""A dictionary that represent a data request, like MARS, CDS, OpenData, ..."""
