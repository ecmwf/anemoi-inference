# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from unittest.mock import patch

import pytest

import anemoi.inference.checkpoint
from anemoi.inference.runner import Runner

from ..metadata.fake_metadata import FakeMetadata


@pytest.fixture(scope="session")
def fake_huggingface_repo(tmp_path_factory):
    """Create a fake huggingface repo download"""
    tmp_dir = tmp_path_factory.mktemp("repo")
    fn = tmp_dir / "model.ckpt"
    fn.write_text("TESTING", encoding="utf-8")
    return tmp_dir


@pytest.fixture(scope="session")
def fake_huggingface_ckpt(tmp_path_factory):
    """Create a fake huggingface ckpt download"""
    tmp_dir = tmp_path_factory.mktemp("repo")
    fn = tmp_dir / "model.ckpt"
    fn.write_text("TESTING", encoding="utf-8")
    return fn


@patch("huggingface_hub.snapshot_download")
@pytest.mark.parametrize("ckpt", ["organisation/test_repo"])
def test_huggingface_repo_download_str(huggingface_mock, monkeypatch, ckpt, fake_huggingface_repo):

    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_repo

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_repo / "model.ckpt")

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(repo_id=ckpt)


@patch("huggingface_hub.snapshot_download")
@pytest.mark.parametrize("ckpt", [{"repo_id": "organisation/test_repo"}])
def test_huggingface_repo_download_dict(huggingface_mock, monkeypatch, ckpt, fake_huggingface_repo):

    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_repo

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_repo / "model.ckpt")

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(**ckpt)


@patch("huggingface_hub.hf_hub_download")
@pytest.mark.parametrize("ckpt", [{"repo_id": "organisation/test_repo", "filename": "model.ckpt"}])
def test_huggingface_file_download(huggingface_mock, monkeypatch, ckpt, fake_huggingface_ckpt):

    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_ckpt

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_ckpt)

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(**ckpt)
