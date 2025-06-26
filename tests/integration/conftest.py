pytest_plugins = "anemoi.utils.testing"


def pytest_configure(config):
    config.option.log_cli = True
    config.option.log_cli_level = "INFO"
