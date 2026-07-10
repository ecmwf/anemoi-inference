# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
from datetime import datetime
from pathlib import Path

import eccodes
import pytest
import yaml
from anemoi.utils.testing import GetTestData
from earthkit.data.readers.grib.codes import GribCodesHandle
from earthkit.data.readers.grib.codes import GribField
from earthkit.data.utils.dates import to_timedelta
from pytest_mock import MockerFixture
from rich import print

from anemoi.inference.grib.encoding import GribWriter
from anemoi.inference.grib.encoding import check_encoding
from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.templates.input import InputTemplates
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.metadata import Metadata
from anemoi.inference.outputs.parallel import _restore_grib_templates
from anemoi.inference.outputs.parallel import _sanitise_state
from anemoi.inference.testing import files_for_tests


def _metadata():
    with open(files_for_tests("unit/checkpoints/atmos.json"), "r") as f:
        data = json.load(f)
    metadata = Metadata(data)
    metadata.typed_variables["unknown"] = metadata.typed_variables["2t"]  # used in auto-unknown test

    # 100u failed at one point in test_builtin_gribwriter, make sure 100u is still present if checkpoint is ever updated
    assert "100u" in metadata.typed_variables.keys()
    return metadata


@pytest.fixture
def manager(mocker: MockerFixture) -> type[TemplateManager]:
    def _manager(config=None):
        owner = mocker.MagicMock()
        owner.metadata = _metadata()
        return TemplateManager(owner, templates=config)

    return _manager


@pytest.fixture
def grib_template(get_test_data: GetTestData) -> Path:
    return get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")


@pytest.fixture
def template_index(grib_template: Path, tmp_path: Path) -> Path:
    index = [
        [{"grid": "O96", "levtype": "pl"}, grib_template],
        [{"grid": "O96"}, grib_template],
    ]
    index_file = tmp_path / "templates_index.yaml"
    with open(index_file, "w") as f:
        yaml.dump(index, f)
    return index_file


@pytest.mark.parametrize(
    "variable, expected_param",
    [
        pytest.param("2t", "lsm", id="sfc"),  # lsm is the builtin template for sfc
        pytest.param("w_100", "q", id="pl"),  # q is the builtin template for pl
    ],
)
def test_builtin(manager, variable, expected_param):
    manager = manager()
    template = manager.template(variable, state={}, typed_variables=manager.typed_variables)

    assert isinstance(template, GribField)
    assert template.metadata("param") == expected_param


@pytest.mark.parametrize(
    "file_config, variable, expected_param, expected_type",
    [
        pytest.param({}, "2t", "10u", GribField, id="first"),
        pytest.param({"mode": "last"}, "2t", "v", GribField, id="last"),
        pytest.param({"mode": "auto"}, "2t", "2t", GribField, id="auto-sfc"),
        pytest.param({"mode": "auto"}, "w_100", "w", GribField, id="auto-pl"),
        pytest.param({"mode": "auto"}, "unknown", None, type(None), id="auto unknown"),
        pytest.param({"variables": "10u"}, "2t", None, type(None), id="skip variable"),
    ],
)
def test_file(manager, grib_template, file_config, variable, expected_param, expected_type):
    config = {
        "file": {
            "path": grib_template,
            **file_config,
        }
    }
    manager = manager(config)
    template = manager.template(variable, state={}, typed_variables=manager.typed_variables)

    assert isinstance(template, expected_type)

    if expected_param is not None:
        assert template.metadata("param") == expected_param


def test_samples_index_path(manager, template_index):
    config = {
        "samples": {"index_path": str(template_index)},
    }
    manager = manager(config)
    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)

    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_samples_index_path_str(manager, template_index):
    config = {
        "samples": str(template_index),
    }
    manager = manager(config)
    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)

    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_samples_direct_index(manager, template_index):
    with open(template_index, "r") as f:
        index = yaml.safe_load(f)
    config = {"samples": index}
    manager = manager(config)
    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)

    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="default"),
        pytest.param("input", id="input only"),
        pytest.param(["input", "builtin"], id="input+builtin"),
        pytest.param({"input": {"10u": "2t", "2t": "ignored"}}, id="input with fallback"),
    ],
)
def test_input(manager, config, request):
    manager = manager(config)
    state = {
        "_grib_templates_for_output": {
            "2t": "2t_input_template",
        }
    }

    template = manager.template("2t", state=state, typed_variables=manager.typed_variables)
    assert template == "2t_input_template"

    template = manager.template("10u", state=state, typed_variables=manager.typed_variables)
    if request.node.callspec.id == "input only":
        assert template is None
    elif request.node.callspec.id == "input with fallback":
        assert template == "2t_input_template"
    else:
        assert template.metadata("param") == "lsm"  # builtin


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(variable, id=variable.param)
        for variable in {
            var.param: var for var in _metadata().typed_variables.values() if not var.is_computed_forcing
        }.values()  # dict to get only one level per pl param
    ],
)
def test_builtin_gribwriter(manager, variable, tmp_path):
    """Test that the builtin grib templates can be encoded and written with new metadata."""

    manager = manager("builtin")
    builtin_provider = manager.templates_providers[0]

    keys = {
        "stream": "oper",
        "type": "fc",
        "class": "od",
        "expver": "9999",
    }

    grids = set()
    for match, _ in builtin_provider.templates:
        if grid := match.get("grid"):
            grids.add(grid)

    for grid in grids:
        lookup = {
            "grid": grid,
            "levtype": "sfc" if variable.is_surface_level else "pl",
        }
        template = builtin_provider.template(None, lookup, state={})
        metadata = grib_keys(
            values=None,
            template=template,
            variable=variable,
            param=variable.param,
            date=datetime.now(),
            step=to_timedelta(6),
            keys=keys,
            ensemble=False,
            start_steps={},
            previous_step=None,
        )
        print(lookup, template, metadata)
        output = GribWriter(tmp_path / f"output-{grid}.grib")
        handle, _ = output.write(values=None, template=template, metadata=metadata)
        check_encoding(handle, metadata)
        output.close()


# ── Parallel output: GRIB template serialisation / reconstruction ────────────


def _make_grib_field_mock(short_name: str, bits_per_value: int) -> object:
    """Create a minimal mock field whose .message() returns real GRIB bytes."""
    raw = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set(raw, "shortName", short_name)
    eccodes.codes_set(raw, "bitsPerValue", bits_per_value)
    msg = eccodes.codes_get_message(raw)
    eccodes.codes_release(raw)

    class _MockField:
        def message(self):
            return msg

    return _MockField()


def test_sanitise_state_serialises_grib_templates():
    """_sanitise_state converts _grib_templates_for_output to picklable bytes."""
    field = _make_grib_field_mock("2t", bits_per_value=16)
    state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}

    sanitised = _sanitise_state(state)

    assert "_grib_templates_for_output" not in sanitised
    assert "_grib_templates_bytes_for_output" in sanitised
    assert isinstance(sanitised["_grib_templates_bytes_for_output"]["2t"], bytes)


def test_sanitise_state_bytes_are_picklable():
    """Bytes produced by _sanitise_state survive a pickle round-trip."""
    import pickle

    field = _make_grib_field_mock("2t", bits_per_value=16)
    state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}
    sanitised = _sanitise_state(state)

    restored = pickle.loads(pickle.dumps(sanitised))
    assert isinstance(restored["_grib_templates_bytes_for_output"]["2t"], bytes)


def test_restore_grib_templates_reconstructs_wrapper(mocker: MockerFixture):
    """_restore_grib_templates turns serialised bytes back into a real GribField,
    which InputTemplates can then serve without any bytes-awareness of its own.
    """
    field = _make_grib_field_mock("2t", bits_per_value=16)
    state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}
    sanitised = _sanitise_state(state)
    restored = _restore_grib_templates(sanitised)

    assert "_grib_templates_bytes_for_output" not in restored
    assert "_grib_templates_for_output" in restored

    owner = mocker.MagicMock()
    owner.metadata = _metadata()
    provider = InputTemplates(owner)

    template = provider.template("2t", {}, state=restored)

    assert isinstance(template, GribField)
    assert template.metadata("shortName") == "2t"
    assert template.metadata("bitsPerValue") == 16


def test_restore_grib_templates_preserves_bits_per_value(mocker: MockerFixture):
    """Reconstructed template preserves the original bitsPerValue so packing is identical."""
    for bpv in (16, 24):
        field = _make_grib_field_mock("2t", bits_per_value=bpv)
        state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}
        restored = _restore_grib_templates(_sanitise_state(state))

        owner = mocker.MagicMock()
        owner.metadata = _metadata()
        provider = InputTemplates(owner)

        template = provider.template("2t", {}, state=restored)
        assert template.metadata("bitsPerValue") == bpv, f"expected bpv={bpv}"


def test_restore_grib_templates_noop_without_bytes():
    """_restore_grib_templates leaves states without serialised templates untouched."""
    state = {"fields": {}}
    restored = _restore_grib_templates(state)
    assert "_grib_templates_for_output" not in restored
    assert "_grib_templates_bytes_for_output" not in restored


def test_restore_grib_templates_populates_cache():
    """After restore, _grib_templates_for_output is populated on the state."""
    field = _make_grib_field_mock("2t", bits_per_value=16)
    state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}
    restored = _restore_grib_templates(_sanitise_state(state))

    assert "2t" in restored.get("_grib_templates_for_output", {})


def test_restore_grib_templates_handle_is_cloneable():
    """The restored field's handle can be cloned — required by encode_message."""
    field = _make_grib_field_mock("2t", bits_per_value=16)
    state = {"_grib_templates_for_output": {"2t": field}, "fields": {}}
    restored = _restore_grib_templates(_sanitise_state(state))

    template = restored["_grib_templates_for_output"]["2t"]
    cloned = template.handle.clone()

    assert isinstance(cloned, GribCodesHandle)
    assert eccodes.codes_get(cloned._handle, "bitsPerValue") == 16
