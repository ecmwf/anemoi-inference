def pytest_configure(config):
    config.option.log_cli = False
    config.option.log_cli_level = "INFO"
