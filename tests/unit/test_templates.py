from pathlib import Path

import yaml
from anemoi.utils.testing import GetTestData
from earthkit.data.readers.grib.codes import GribField
from pytest_mock import MockerFixture

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.testing import fake_checkpoints


@fake_checkpoints
def test_manager_builtin(mocker: MockerFixture):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    manager = TemplateManager(owner)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "lsm"  # lsm is used as the builtin template for surface fields


@fake_checkpoints
def test_manager_file(mocker: MockerFixture, get_test_data: GetTestData):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")

    config = {"file": template_file}

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


@fake_checkpoints
def test_manager_file_last(mocker: MockerFixture, get_test_data: GetTestData):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")

    config = {"file": {"path": template_file, "mode": "last"}}

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "v"  # last field in the file


@fake_checkpoints
def test_manager_file_skip_variable(mocker: MockerFixture, get_test_data: GetTestData):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")

    config = {"file": {"path": template_file, "variable": "10u"}}

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert template is None


@fake_checkpoints
def test_manager_file_auto(mocker: MockerFixture, get_test_data: GetTestData):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")

    config = {"file": {"path": template_file, "mode": "auto"}}

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "2t"


@fake_checkpoints
def test_manager_file_auto_unknown(mocker: MockerFixture, get_test_data: GetTestData):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")

    config = {"file": {"path": template_file, "mode": "auto"}}

    manager = TemplateManager(owner, templates=config)
    typed = c.typed_variables
    typed["unknown"] = typed["2t"]

    template = manager.template("unknown", state={}, typed_variables=typed)
    assert template is None


@fake_checkpoints
def test_manager_samples(mocker: MockerFixture, get_test_data: GetTestData, tmp_path: Path):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

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

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


@fake_checkpoints
def test_manager_samples_index_str(mocker: MockerFixture, get_test_data: GetTestData, tmp_path: Path):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

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

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


@fake_checkpoints
def test_manager_samples_direct_index(mocker: MockerFixture, get_test_data: GetTestData, tmp_path: Path):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    template_file = get_test_data("anemoi-integration-tests/inference/single-o48-1.1/input.grib")
    index = [
        [{"grid": "O96", "levtype": "pl"}, template_file],
        [{"grid": "O96"}, template_file],
    ]

    config = {"samples": index}

    manager = TemplateManager(owner, templates=config)

    template = manager.template("2t", state={}, typed_variables=c.typed_variables)
    assert isinstance(template, GribField)
    assert template.metadata("param") == "10u"  # first field in the file


@fake_checkpoints
def test_manager_input(mocker: MockerFixture):
    c = Checkpoint("unit/checkpoints/simple.ckpt")
    owner = mocker.MagicMock()
    owner.context.checkpoint = c

    manager = TemplateManager(owner)
    state = {
        "_grib_templates_for_output": {
            "2t": "2t_input_template",
        }
    }

    template = manager.template("2t", state=state, typed_variables=c.typed_variables)
    assert template == "2t_input_template"
