# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
from typing import Any
from typing import Union

from numpy.typing import NDArray

"""A collection of types used in the inference module.
Some of these type could be moved to anemoi.utils.types or anemoi.transform.types.
"""

State = dict[str, Any]
"""A dictionary that represents the state of a model."""

DataRequest = dict[str, Any]
"""A dictionary that represent a data request, like MARS, CDS, OpenData, ..."""

Date = Union[str, datetime.datetime, int]
"""A date can be a string, a datetime object or an integer. It will always be converted to a datetime object."""

IntArray = NDArray[Any]
"""A numpy array of integers."""

FloatArray = NDArray[Any]
"""A numpy array of floats."""

BoolArray = NDArray[Any]
"""A numpy array of booleans."""

Shape = tuple[int, ...]
"""A tuple of integers representing the shape of an array."""

ProcessorConfig = Union[str, dict[str, Any]]
"""A str or dict of str representing a pre- or post-processor configuration."""
