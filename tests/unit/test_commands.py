# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``inspect`` and ``patch`` commands on the test checkpoints."""

import json
from argparse import Namespace
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from anemoi.inference.commands.inspect import InspectCmd
from anemoi.inference.commands.patch import PatchCmd
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata

LEGACY = "unit/checkpoints/atmos.json"
MULTI = "unit/checkpoints/multi-single.json"


def _inspect_args(path, **kwargs):
    args = dict(path=path, variables=False, requirements=False, datasets=False, dump=False, json=False, validate=False)
    args.update(kwargs)
    return Namespace(**args)


@fake_checkpoints
def test_inspect_variables_legacy(capsys):
    InspectCmd().run(_inspect_args(LEGACY, variables=True))
    out = capsys.readouterr().out
    assert "Dataset:" not in out
    assert "2t" in out and "prognostic" in out


@fake_checkpoints
def test_inspect_variables_multi_dataset_names_the_dataset(capsys):
    InspectCmd().run(_inspect_args(MULTI, variables=True))
    out = capsys.readouterr().out
    assert "Dataset: data" in out
    assert "prognostic" in out


@fake_checkpoints
def test_inspect_dump_json(capsys):
    InspectCmd().run(_inspect_args(LEGACY, dump=True, json=True))
    dumped = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert dumped["multi_step_input"] == 2
    assert "2t" in dumped["typed_variables"]


@fake_checkpoints
def test_inspect_datasets(capsys):
    InspectCmd().run(_inspect_args(LEGACY, datasets=True))
    assert "Open dataset arguments:" in capsys.readouterr().out


@fake_checkpoints
def test_inspect_requirements(capsys):
    InspectCmd().run(_inspect_args(LEGACY, requirements=True))
    assert "automatically generated from a checkpoint" in capsys.readouterr().out


def _stored(path, arrays=None):
    metadata, _ = mock_load_metadata(path, supporting_arrays=True)
    return deepcopy(metadata), deepcopy(arrays if arrays is not None else {})


def _run_patch(path, new, stored_arrays=None, **kwargs):
    """Run the patch command with *new* as the metadata read from the training dataset."""
    written = []

    def fake_replace(path, metadata, supporting_arrays):
        written.append((metadata, supporting_arrays))

    def fake_load(path, supporting_arrays=False):
        metadata, arrays = _stored(LEGACY, stored_arrays)
        return (metadata, arrays) if supporting_arrays else metadata

    with (
        patch("anemoi.inference.checkpoint.load_metadata", fake_load),
        patch("anemoi.inference.checkpoint.Checkpoint.update_metadata_from_zarr", return_value=new),
        patch("anemoi.utils.checkpoints.replace_metadata", fake_replace),
    ):
        PatchCmd().run(Namespace(path=path, sanitise=False, **kwargs))
    return written


def test_patch_does_nothing_when_up_to_date():
    assert _run_patch(LEGACY, _stored(LEGACY)) == []


def test_patch_does_nothing_when_arrays_are_up_to_date():
    arrays = {"latitudes": np.arange(3.0)}
    assert _run_patch(LEGACY, _stored(LEGACY, arrays), stored_arrays=arrays) == []


def test_patch_writes_changed_metadata():
    metadata, arrays = _stored(LEGACY)
    metadata["dataset"]["variables"] = [*metadata["dataset"]["variables"], "extra"]
    (written_metadata, written_arrays), *rest = _run_patch(LEGACY, (metadata, arrays))
    assert not rest
    assert written_metadata["dataset"]["variables"][-1] == "extra"
    assert written_arrays == {}


def test_patch_writes_changed_arrays():
    stored = {"latitudes": np.arange(3.0)}
    new = {"latitudes": np.arange(3.0) + 1}
    metadata, _ = _stored(LEGACY)
    (_, written_arrays), *rest = _run_patch(LEGACY, (metadata, new), stored_arrays=stored)
    assert not rest
    np.testing.assert_array_equal(written_arrays["latitudes"], new["latitudes"])


@pytest.mark.parametrize("path", [LEGACY, MULTI])
@fake_checkpoints
def test_inspect_validate_runs(path):
    InspectCmd().run(_inspect_args(path, validate=True))
