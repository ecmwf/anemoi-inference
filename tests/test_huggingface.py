# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

import anemoi.inference.checkpoint
from anemoi.inference.runner import Runner

from .fake_metadata import FakeMetadata


@pytest.fixture(scope="session")
def fake_huggingface_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a fake huggingface repo download.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary directories.

    Returns
    -------
    Path
        Path to the fake huggingface repo.
    """
    tmp_dir = tmp_path_factory.mktemp("repo")
    fn = tmp_dir / "model.ckpt"
    fn.write_text("TESTING", encoding="utf-8")
    return tmp_dir


@pytest.fixture(scope="session")
def fake_huggingface_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a fake huggingface ckpt download.

        Factory for temporary directories.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary directories

    Returns
    -------
    pathlib.Path
        Path to the fake huggingface checkpoint.
    """
    tmp_dir = tmp_path_factory.mktemp("repo")
    fn = tmp_dir / "model.ckpt"
    fn.write_text("TESTING", encoding="utf-8")
    return fn


@patch("huggingface_hub.snapshot_download")
@pytest.mark.parametrize("ckpt", ["organisation/test_repo"])
def test_huggingface_repo_download_str(
    huggingface_mock: unittest.mock.Mock,
    monkeypatch: pytest.MonkeyPatch,
    ckpt: str,
    fake_huggingface_repo: Path,
) -> None:
    """Test downloading a huggingface repo using a string identifier.

    Parameters
    ----------
    huggingface_mock : unittest.mock.Mock
        Mock for the huggingface snapshot download.
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture for modifying attributes.
    ckpt : str
        Checkpoint identifier.
    fake_huggingface_repo : pathlib.Path
        Path to the fake huggingface repo.
    """
    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_repo

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_repo / "model.ckpt")

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(repo_id=ckpt)


@patch("huggingface_hub.snapshot_download")
@pytest.mark.parametrize("ckpt", [{"repo_id": "organisation/test_repo"}])
def test_huggingface_repo_download_dict(
    huggingface_mock: unittest.mock.Mock,
    monkeypatch: pytest.MonkeyPatch,
    ckpt: Dict[str, str],
    fake_huggingface_repo: Path,
) -> None:
    """Test downloading a huggingface repo using a dictionary identifier.

    Parameters
    ----------
    huggingface_mock : unittest.mock.Mock
        Mock for the huggingface snapshot download.
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture for modifying attributes.
    ckpt : Dict[str, str]
        Checkpoint identifier.
    fake_huggingface_repo : pathlib.Path
        Path to the fake huggingface repo.
    """
    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_repo

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_repo / "model.ckpt")

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(**ckpt)


@patch("huggingface_hub.hf_hub_download")
@pytest.mark.parametrize("ckpt", [{"repo_id": "organisation/test_repo", "filename": "model.ckpt"}])
def test_huggingface_file_download(
    huggingface_mock: unittest.mock.Mock,
    monkeypatch: pytest.MonkeyPatch,
    ckpt: Dict[str, str],
    fake_huggingface_ckpt: Path,
) -> None:
    """Test downloading a specific file from a huggingface repo.

    Parameters
    ----------
    huggingface_mock : unittest.mock.Mock
        Mock for the huggingface file download.
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture for modifying attributes.
    ckpt : Dict[str, str]
        Checkpoint identifier.
    fake_huggingface_ckpt : pathlib.Path
        Path to the fake huggingface checkpoint.
    """
    monkeypatch.setattr(anemoi.inference.checkpoint.Checkpoint, "_metadata", FakeMetadata())
    huggingface_mock.return_value = fake_huggingface_ckpt

    runner = Runner({"huggingface": ckpt})
    assert runner.checkpoint.path == str(fake_huggingface_ckpt)

    assert huggingface_mock.called
    huggingface_mock.assert_called_once_with(**ckpt)
