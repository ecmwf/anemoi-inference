# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def fake_checkpoints(func):
    def wrapper(*args, **kwargs):
        from unittest.mock import patch

        from .mock_checkpoint import mock_load_metadata

        with patch("anemoi.inference.checkpoint.load_metadata", mock_load_metadata):
            with patch("anemoi.inference.metadata.USE_LEGACY", False):
                return func(*args, **kwargs)

    return wrapper
