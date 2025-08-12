# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from .protocol import MetadataProtocol

LOG = logging.getLogger(__name__)


@contextmanager
def patch_function(target: Any, attribute: str, replacement: Any) -> Generator[None, None, None]:
    """Context manager to temporarily replace an attribute of a target object.

    Parameters
    ----------
    target : object
        The target object whose attribute will be replaced.
    attribute : str
        The name of the attribute to replace.
    replacement : any
        The replacement value for the attribute.

    Returns
    -------
    Generator[None, None, None]
        The context manager.
    """
    original = getattr(target, attribute)
    setattr(target, attribute, replacement)
    try:
        yield
    finally:
        setattr(target, attribute, original)


class PatchMixin(MetadataProtocol):

    # `self` is a `Metadata` object

    def patch_metadata(
        self,
        supporting_arrays: dict[str, Any],
        root: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Patch the metadata with supporting arrays and root.

        Parameters
        ----------
        supporting_arrays : dict
            The supporting arrays to patch.
        root : str
            The root path for the supporting arrays.

        Returns
        -------
        tuple
            The patched metadata and supporting arrays.
        """

        metadata, supporting_arrays = self._from_zarr
        self._metadata["dataset"] = metadata
        self._supporting_arrays = supporting_arrays
        return self._metadata, self._supporting_arrays

    @cached_property
    def _from_zarr(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Open the dataset and fetch metadata and supporting arrays."""
        # We assume that the datasets are reachable via the content of
        # ~/.config/anemoi/settings.toml.

        ds = self.open_dataset()
        return ds.metadata(), ds.supporting_arrays()
