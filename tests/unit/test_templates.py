# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from anemoi.utils.testing import GetTestData
from earthkit.data.readers.grib.codes import GribField
from earthkit.data.utils.dates import to_timedelta
from pytest_mock import MockerFixture
from rich import print

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.grib.encoding import GribWriter
from anemoi.inference.grib.encoding import check_encoding
from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


@fake_checkpoints
def _checkpoint():
    checkpoint = Checkpoint(files_for_tests("unit/checkpoints/atmos.json"))
    checkpoint.typed_variables["unknown"] = checkpoint.typed_variables["2t"]  # used in auto-unknown test

    # 100u failed at one point in test_builtin_gribwriter, make sure 100u is still present if checkpoint is ever updated
    assert "100u" in checkpoint.typed_variables.keys()
    return checkpoint


@pytest.fixture
def manager(mocker: MockerFixture) -> type[TemplateManager]:
    @fake_checkpoints
    def _manager(config=None):
        owner = mocker.MagicMock()
        owner.context.checkpoint = _checkpoint()
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
            var.param: var for var in _checkpoint().typed_variables.values() if not var.is_computed_forcing
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
