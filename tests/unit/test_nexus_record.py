# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import json

import pytest

nexus_record = importlib.import_module("anemoi.inference.commands.nexus-record")


def test_checkpoint_nexus_record(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"x" * 10)
    monkeypatch.setattr(
        "anemoi.utils.checkpoints.load_metadata",
        lambda path: {"uuid": "u1", "version": "1.0", "dataset": {}},
    )
    record = nexus_record.checkpoint_nexus_record(str(ckpt), {"projects": ["MLP"], "licenses": ["X"]})
    assert record == {
        "uuid": "u1",
        "metadata": {"uuid": "u1", "version": "1.0", "dataset": {}, "size": 10},
        "projects": ["MLP"],
        "licenses": ["X"],
    }
    monkeypatch.setattr("anemoi.utils.checkpoints.load_metadata", lambda path: {})
    with pytest.raises(ValueError, match="no 'uuid'"):
        nexus_record.checkpoint_nexus_record(str(ckpt))


def test_nexus_record_command(tmp_path, capsys, monkeypatch) -> None:
    from anemoi.inference.__main__ import main

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"abc")
    monkeypatch.setattr("anemoi.utils.checkpoints.load_metadata", lambda path: {"uuid": "u2"})
    attrs = tmp_path / "attrs.json"
    attrs.write_text(json.dumps({"owner": "alice", "license": "CC-BY-4.0"}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "anemoi-inference",
            "nexus-record",
            str(ckpt),
            f"@{attrs}",
            "--project",
            "MLP",
            "--name",
            "my-model",
        ],
    )
    with pytest.raises(SystemExit) as exit:
        main()
    assert exit.value.code in (0, None)
    record = json.loads(capsys.readouterr().out)
    assert record == {
        "uuid": "u2",
        "name": "my-model",
        "metadata": {"uuid": "u2", "size": 3},
        "owner": "alice",
        "licenses": ["CC-BY-4.0"],
        "projects": ["MLP"],
    }
