import pytest
from pytest import MonkeyPatch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.testing import files_for_tests


def test_load_config() -> None:
    """Test loading the configuration"""
    RunConfiguration.load(files_for_tests("unit/configs/simple.yaml"))


def test_interpolation(monkeypatch: MonkeyPatch) -> None:
    """Test loading the configuration with some OmegaConf interpolations"""
    monkeypatch.setenv("TEST", "foo")
    config = RunConfiguration.load(files_for_tests("unit/configs/interpolation.yaml"))
    assert config.name == "foo"
    assert config.name == config.description


def test_config_dotlist_override() -> None:
    """Test overriding the configuration with some dotlist parameters"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/mwd.yaml"),
        overrides=[
            "runner=testing",
            "device=cpu",
            "input=dummy",
            "post_processors.0.backward_transform_filter=test",
        ],
    )
    assert config.post_processors is not None and len(config.post_processors) == 1
    assert isinstance(config.post_processors[0], dict)
    assert config.post_processors[0]["backward_transform_filter"] == "test"


def test_config_dotlist_override_append() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/mwd.yaml"),
        overrides=[
            "runner=testing",
            "device=cpu",
            "input=dummy",
            "post_processors.1.backward_transform_filter=test",
        ],
    )
    assert config.post_processors is not None and len(config.post_processors) == 2
    assert isinstance(config.post_processors[1], dict)
    assert config.post_processors[1]["backward_transform_filter"] == "test"


def test_config_dotlist_override_append_end() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=[
            "pre_processors=[]",
            "pre_processors.0=test",
        ],
    )
    assert isinstance(config.pre_processors, list) and len(config.pre_processors) == 1
    assert config.pre_processors[0] == "test"


def test_config_dotlist_override_add_new_non_empty_dict() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=["input={grib: test}"],
    )
    assert isinstance(config.input, dict) and "grib" in config.input
    assert config.input["grib"] == "test"


def test_config_dotlist_override_add_new_non_empty_list() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=["pre_processors=[test]"],
    )
    assert isinstance(config.pre_processors, list) and len(config.pre_processors) == 1
    assert config.pre_processors[0] == "test"


def test_config_dotlist_override_index_error() -> None:
    """Test failing with index error in dotlist argument"""
    with pytest.raises(IndexError):
        RunConfiguration.load(
            files_for_tests("unit/configs/mwd.yaml"),
            overrides=[
                "runner=testing",
                "device=cpu",
                "input=dummy",
                "post_processors.2.backward_transform_filter=test",
            ],
        )
