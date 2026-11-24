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
import sys
from pathlib import Path

REFGEN = Path(__file__).resolve().parents[2] / "docs" / "config_reference" / "refgen.py"


def test_docs_config_reference_matches_code() -> None:
    """Specs and examples agree with the code, and draft specs are up to date. See ``refgen.py check``."""
    spec = importlib.util.spec_from_file_location("refgen", REFGEN)
    refgen = importlib.util.module_from_spec(spec)
    sys.modules["refgen"] = refgen
    spec.loader.exec_module(refgen)

    assert refgen.check_all() == []
    assert refgen.generate(write=False) == []
