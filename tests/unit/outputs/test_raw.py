# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging

import numpy as np
import pytest

from anemoi.inference.outputs.raw import RawOutput

LOG = logging.getLogger(__name__)

FAKE_PROVENANCE = {"module_versions": {}}


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch):
    """Stub the environment scan, which otherwise dominates the runtime of these tests."""
    monkeypatch.setattr("anemoi.inference.outputs.raw.gather_provenance_info", lambda: FAKE_PROVENANCE)


@pytest.mark.parametrize(
    "variables, expected_fields, not_expected_fields",
    [
        pytest.param(None, ["z_500", "cp", "2t"], [], id="none_variables"),
        pytest.param({"select": ["z_500", "cp"]}, ["z_500", "cp"], ["2t"], id="select_list"),
        pytest.param({"drop": "cp"}, ["z_500", "2t"], ["cp"], id="drop_single_string"),
    ],
)
def test_raw_output_write_step(
    variables, expected_fields, not_expected_fields, basic_context, basic_metadata, basic_state, tmp_path
):
    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path), variables=variables)
    output.write_step(basic_state)

    written = tmp_path / "20200101000000.npz"
    assert written.exists()

    with np.load(written) as data:
        for variable in expected_fields:
            assert f"field_{variable}" in data.files
        for variable in not_expected_fields:
            assert f"field_{variable}" not in data.files
        assert {"date", "latitudes", "longitudes"} <= set(data.files)


@pytest.mark.parametrize("with_grid", [True, False], ids=["with_grid", "without_grid"])
def test_raw_output_write_manifest(with_grid, basic_context, basic_metadata, basic_state, tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    basic_context.checkpoint.path = str(checkpoint)

    basic_metadata.grid = "o96"

    state = dict(basic_state)
    if not with_grid:
        del state["latitudes"], state["longitudes"]

    dir = tmp_path / "output"
    output = RawOutput(basic_context, basic_metadata, dir=str(dir), variables={"drop": "cp"}, output_manifest=True)
    output.open(state)

    manifest = json.loads((dir / "manifest.json").read_text())

    assert manifest["variables"] == ["z_500", "2t"]
    assert manifest["dataset_name"] == "test"
    assert manifest["grid"] == "o96"
    assert manifest["template"] == "{date}.npz"
    assert manifest["checkpoint"]["path"] == str(checkpoint)
    assert manifest["checkpoint"]["md5"] is not None
    assert manifest["provenance"] == FAKE_PROVENANCE

    assert (dir / "grid.npz").exists() is with_grid


def test_raw_output_manifest_disabled(basic_context, basic_metadata, basic_state, tmp_path):
    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path))
    output.open(basic_state)

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "grid.npz").exists()
