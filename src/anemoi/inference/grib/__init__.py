# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Helpers shared by the GRIB encoding and output code."""

from typing import Any


def grib_handle(field: Any) -> Any:
    """Return the eccodes handle backing a GRIB field.

    earthkit-data 1.0 no longer exposes ``Field.handle``. The handle is instead
    available as the ``handle`` GRIB metadata key (one of earthkit-data's
    ``CUSTOM_KEYS``), i.e. ``field.metadata("handle")``.

    Objects that still carry a ``handle`` attribute directly, such as the template
    wrappers used internally and in the tests, are supported too.

    Parameters
    ----------
    field : Any
        The field, or handle-carrying wrapper, to read the handle from.

    Returns
    -------
    Any
        The eccodes handle.

    Raises
    ------
    ValueError
        If the field is not backed by a GRIB message.
    """
    handle = getattr(field, "handle", None)
    if handle is not None:
        return handle

    get = getattr(field, "get", None)
    if get is not None:
        # Returns None rather than raising when the field is not GRIB-backed.
        handle = get("metadata.handle", default=None)
        if handle is not None:
            return handle

    raise ValueError(f"Not a GRIB field, cannot access its GRIB handle: {field!r}")
