# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any


class FakeMetadata:
    """A class to simulate metadata for testing purposes.

    This class returns None for any attribute accessed, simulating the absence
    of metadata.
    """

    def __getattr__(self, name: str) -> Any:
        """Simulate the absence of metadata by returning None for any attribute.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            Always returns None.
        """
        return None
