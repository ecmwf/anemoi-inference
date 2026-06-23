# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from types import SimpleNamespace

import pytest

from anemoi.inference.inputs.dataset import DatasetInput

_CTX = SimpleNamespace(verbosity=0, reference_date=None)
_META = SimpleNamespace(dataset_name="test_dataset", _supporting_arrays={})
_KWARGS = dict(variables=[], open_dataset_args=(), open_dataset_kwargs={})


def test_warning_logged_when_pre_processors_configured(caplog):
    with caplog.at_level(logging.WARNING, logger="anemoi.inference.inputs.dataset"):
        DatasetInput(_CTX, _META, pre_processors=[{"scale": {}}], **_KWARGS)
    assert any("will NOT be applied" in r.message for r in caplog.records)


def test_no_warning_when_no_pre_processors(caplog):
    with caplog.at_level(logging.WARNING, logger="anemoi.inference.inputs.dataset"):
        DatasetInput(_CTX, _META, **_KWARGS)
    assert not any("will NOT be applied" in r.message for r in caplog.records)


def test_uses_warn_once_when_available():
    warned = []
    ctx = SimpleNamespace(verbosity=0, reference_date=None, _warn_once=warned.append)
    DatasetInput(ctx, _META, pre_processors=[{"scale": {}}], **_KWARGS)
    assert len(warned) == 1
    assert "will NOT be applied" in warned[0]
