from pathlib import Path

import pytest
import yaml
from anemoi.utils.testing import GetTestData
from earthkit.data.readers.grib.codes import GribField
from pytest_mock import MockerFixture

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


@pytest.fixture
def manager(mocker: MockerFixture) -> type[TemplateManager]:
    owner = mocker.MagicMock()
    owner.context.checkpoint = Checkpoint(files_for_tests("unit/checkpoints/simple.yaml"))

    @fake_checkpoints
    def _manager(config=None):
        return TemplateManager(owner, templates=config)

    return _manager


def test_builtin(manager):
    manager = manager()
    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "lsm"  # lsm is used as the builtin template for surface fields


def test_file(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    config = {"file": template_file}

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_file_last(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    config = {"file": {"path": template_file, "mode": "last"}}

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "v"  # last field in the file


def test_file_skip_variable(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    config = {"file": {"path": template_file, "variable": "10u"}}

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert template is None


def test_file_auto(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    config = {"file": {"path": template_file, "mode": "auto"}}

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "2t"


def test_file_auto_unknown(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    config = {"file": {"path": template_file, "mode": "auto"}}

    manager = manager(config)
    typed = manager.typed_variables
    typed["unknown"] = typed["2t"]

    template = manager.template("unknown", state={}, typed_variables=typed)
    assert template is None


def test_samples(manager, get_test_data: GetTestData, tmp_path: Path):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    index = [
        [{"grid": "O96", "levtype": "pl"}, template_file],
        [{"grid": "O96"}, template_file],
    ]
    index_file = tmp_path / "templates_index.yaml"
    with open(index_file, "w") as f:
        yaml.dump(index, f)

    config = {
        "samples": {"index_path": str(index_file)},
    }

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_samples_index_str(manager, get_test_data: GetTestData, tmp_path: Path):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    index = [
        [{"grid": "O96", "levtype": "pl"}, template_file],
        [{"grid": "O96"}, template_file],
    ]
    index_file = tmp_path / "templates_index.yaml"
    with open(index_file, "w") as f:
        yaml.dump(index, f)

    config = {
        "samples": str(index_file),
    }

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_samples_direct_index(manager, get_test_data: GetTestData):
    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    index = [
        [{"grid": "O96", "levtype": "pl"}, template_file],
        [{"grid": "O96"}, template_file],
    ]

    config = {"samples": index}

    manager = manager(config)

    template = manager.template("2t", state={}, typed_variables=manager.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


def test_input(manager):
    manager = manager()
    state = {
        "_grib_templates_for_output": {
            "2t": "2t_input_template",
        }
    }

    template = manager.template("2t", state=state, typed_variables=manager.typed_variables)
    assert template == "2t_input_template"
