# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Check the configuration reference in ``docs/config_reference`` against the code."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "config_reference"
NODE = os.environ.get("ANEMOI_NODE") or shutil.which("node")


def _load_refgen() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refgen", REFERENCE_DIR / "refgen.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["refgen"] = module
    spec.loader.exec_module(module)
    return module


refgen = _load_refgen()
SPECS = refgen.load_specs()
EXAMPLES = refgen.load_examples()


@pytest.mark.parametrize("kind", refgen.documented_kinds())
def test_config_reference_covers_registry(kind: str) -> None:
    """Every registered component is documented or explicitly excluded in ``index.yaml``."""
    assert refgen.check_coverage(kind, SPECS) == []


@pytest.mark.parametrize(
    "kind, key",
    [(kind, key) for kind, entries in SPECS.items() for key in entries],
    ids=lambda v: str(v),
)
def test_config_reference_spec_matches_code(kind: str, key: str) -> None:
    """Documented fields, required flags and defaults agree with the constructor signatures."""
    assert refgen.check_spec(kind, key, SPECS) == []


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.path)
def test_config_reference_example_is_valid(example: "refgen.Example") -> None:
    """Every example loads as a run configuration and uses only documented components and fields."""
    errors, _ = refgen.check_example(example, SPECS)
    assert errors == []


def test_config_reference_every_component_has_an_example() -> None:
    """Every documented component appears in at least one example."""
    used = set()
    for example in EXAMPLES:
        _, usages = refgen.check_example(example, SPECS)
        used |= {(u.kind, u.name) for u in usages}
    missing = [
        spec["_path"]
        for kind, entries in SPECS.items()
        for key, spec in entries.items()
        if kind != refgen.TOP_LEVEL_KIND and not key.startswith("_") and (kind, spec["name"]) not in used
    ]
    assert missing == []


def test_config_reference_generated_files_up_to_date() -> None:
    """Draft specs and generated examples match the code. Fix with ``refgen.py generate``."""
    assert refgen.generate(write=False) == []


def test_config_reference_builds(tmp_path: Path) -> None:
    """The page builds into a single self-contained HTML file."""
    page = refgen.build(tmp_path).read_text()
    assert "__DATA__" not in page and "/*__JS__*/" not in page
    assert "<script src=" not in page and "<link " not in page


def _plain(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


@pytest.mark.skipif(NODE is None, reason="needs Node.js (on PATH or set ANEMOI_NODE)")
def test_config_reference_page_logic(tmp_path: Path) -> None:
    """The page renders every route, the builder accepts every component and example, and its YAML parses back."""
    page = refgen.build(tmp_path)
    harness = Path(__file__).with_name("config_reference_page.js")
    result = json.loads(
        subprocess.run([NODE, str(harness), str(page)], capture_output=True, text=True, check=True).stdout
    )

    assert result["errors"] == []
    for case in result["yaml"]:
        assert _plain(yaml.safe_load(case["yaml"])) == _plain(case["data"])
    assert {p["text"]: p["value"] for p in result["parse"]}["0079"] == "0079"
