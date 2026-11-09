# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

zarr = pytest.importorskip("zarr")

from anemoi.inference.outputs.zarr import create_zarr_array  # noqa: E402


@pytest.mark.parametrize("chunks", ["auto", True, False, (1, 2)])
def test_create_zarr_array_chunks(chunks):
    """Every value allowed by the ``chunks`` option of the zarr output must create an array."""
    # ZarrOutput passes a store to zarr 3 and a group to zarr 2
    target = zarr.storage.MemoryStore() if zarr.__version__ >= "3" else zarr.open_group({}, mode="w")

    array = create_zarr_array(target, "x", (4, 2), "f4", ("a", "b"), chunks)

    assert array.shape == (4, 2)
    if chunks is False:
        assert array.chunks == (4, 2)
    elif isinstance(chunks, tuple):
        assert array.chunks == chunks
