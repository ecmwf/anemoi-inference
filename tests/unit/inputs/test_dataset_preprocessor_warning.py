# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from types import SimpleNamespace

from anemoi.inference.inputs.dataset import DatasetInput

_CTX = SimpleNamespace(verbosity=0, reference_date=None)
_META = SimpleNamespace(dataset_name="test_dataset", _supporting_arrays={})
_KWARGS = dict(variables=[], open_dataset_args=(), open_dataset_kwargs={})


def test_warning_logged_when_pre_processors_configured():
    with pytest.warns(UserWarning, match="will NOT be applied"):
        DatasetInput(_CTX, _META, pre_processors=[{"scale": {}}], **_KWARGS)


def test_no_warning_when_no_pre_processors(recwarn):
    DatasetInput(_CTX, _META, **_KWARGS)
    assert not any("will NOT be applied" in str(w.message) for w in recwarn.list)


def test_uses_warn_once_when_available():
    call_count = [0]

    def mock_warn_once(msg):
        call_count[0] += 1
        warnings.warn(msg, UserWarning)

    ctx = SimpleNamespace(verbosity=0, reference_date=None, _warn_once=mock_warn_once)
    with pytest.warns(UserWarning, match="will NOT be applied"):
        DatasetInput(ctx, _META, pre_processors=[{"scale": {}}], **_KWARGS)
    assert call_count[0] == 1
