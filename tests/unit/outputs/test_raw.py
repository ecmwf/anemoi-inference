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
import zipfile

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
    output = RawOutput(basic_context, basic_metadata, dir=str(dir), variables={"drop": "cp"})
    output.open(state)
    output.close()

    manifest = json.loads((dir / "manifest.json").read_text())

    assert manifest["variables"] == ["z_500", "2t"]
    assert manifest["dataset_name"] == "test"
    assert manifest["grid"] == "o96"
    assert manifest["template"] == "{date}.npz"
    assert manifest["checkpoint"]["path"] == str(checkpoint)
    assert manifest["checkpoint"]["md5"] is not None
    assert manifest["provenance"] == FAKE_PROVENANCE


@pytest.mark.parametrize("compressed", [True, False], ids=["compressed", "uncompressed"])
def test_raw_output_atomic_roundtrip(compressed, basic_context, basic_metadata, basic_state, tmp_path, monkeypatch):
    savez = np.savez_compressed if compressed else np.savez
    seen_during_write = []

    def watched_savez(file, **arrays):
        savez(file, **arrays)
        seen_during_write.extend(p.name for p in tmp_path.glob("*.npz"))

    monkeypatch.setattr(np, "savez_compressed" if compressed else "savez", watched_savez)
    rng = np.random.default_rng(0)
    state = dict(
        basic_state, fields={name: rng.standard_normal(16, dtype=np.float32) for name in basic_state["fields"]}
    )

    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path), compressed=compressed)
    output.open(state)
    output.write_step(state)

    # Nothing is visible under a final name while the file is written, and no temporary file is left behind.
    assert seen_during_write == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["20200101000000.npz", "manifest.json.part"]

    written = tmp_path / "20200101000000.npz"
    with zipfile.ZipFile(written) as archive:
        stored = {info.compress_type for info in archive.infolist()}
    assert stored == ({zipfile.ZIP_DEFLATED} if compressed else {zipfile.ZIP_STORED})

    with np.load(written) as data:
        for name, values in state["fields"].items():
            assert data[f"field_{name}"].dtype == values.dtype
            np.testing.assert_array_equal(data[f"field_{name}"], values)


def test_raw_output_failed_write_leaves_no_file(basic_context, basic_metadata, basic_state, tmp_path, monkeypatch):
    def failing_savez(file, **arrays):
        file.write(b"partial")
        assert list(tmp_path.glob("*.npz")) == []
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", failing_savez)
    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path))

    with pytest.raises(OSError, match="disk full"):
        output.write_step(basic_state)

    assert list(tmp_path.iterdir()) == []


def test_raw_output_partial_manifest(basic_context, basic_metadata, basic_state, tmp_path):
    final = tmp_path / "manifest.json"
    partial = tmp_path / "manifest.json.part"
    final.write_text("{}")  # left by a previous run

    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path))
    output.open(basic_state)
    assert not final.exists() and partial.exists()

    output.write_step(basic_state)
    output.close()
    assert final.exists() and not partial.exists()
    assert json.loads(final.read_text())["format"] == "anemoi-raw"

    # Reopening the same instance starts a new run.
    output.open(basic_state)
    assert not final.exists() and partial.exists()
    output.close()
    assert final.exists() and not partial.exists()
