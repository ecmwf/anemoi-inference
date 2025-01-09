# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [Unreleased]

### Added
- Add support for models with unconnected nodes dropped from input [#95](https://github.com/ecmwf/anemoi-inference/pull/95).
- Change trigger for boundary forcings [#95](https://github.com/ecmwf/anemoi-inference/pull/95).
- Add support for automatic loading of anemoi-datasets of more general type [#95](https://github.com/ecmwf/anemoi-inference/pull/95).
- Add initial state output in netcdf format
- Fix: Enable inference when no constant forcings are used
- Add anemoi-transform link to documentation
- Add support for unstructured grids
- Add CONTRIBUTORS.md file (#36)
- Add sanetise command
- Add support for huggingface

### Changed
- Change `write_initial_state` default value to `true`
- Raw output of initial state contains only values at initial time
- Change default naming of raw output
- Add cos_solar_zenith_angle to list of known forcings
- Add missing classes in checkpoint handling
- Rename Condition to State [#24](https://github.com/ecmwf/anemoi-inference/pull/24)
- Fix pre-commit regex
- Complete refactoring
- ci: extend python versions to include 3.11 and 3.12 [#31] (https://github.com/ecmwf/anemoi-inference/pull/31)
- Update copyright notice
- Fix `__version__` import in init
- use earthkit-data 0.11.2
- Fix SimpleRunner

### Removed
- ci: turn off hpc workflow


## [0.2.0](https://github.com/ecmwf/anemoi-inference/compare/0.1.9...0.2.0) - Use earthkit-data

### Added
- ci: changelog release updater
- earthkit-data replaces climetlab
- `validate_environment` on Checkpoint [#13](https://github.com/ecmwf/anemoi-inference/pull/13)
- Validate the environment against a checkpoint with `anemoi-inference inspect --validate path.ckpt`
- ci-hpc-config
- Add Condition to store data [#15](https://github.com/ecmwf/anemoi-inference/pull/15)

### Changed
- Fix: diagnostics bug when fields are non-accumulated, remove diagnostics from mars request [#18](https://github.com/ecmwf/anemoi-inference/pull/18)
- ci: updated workflows on PR and releases to use reusable actions
- removed a variable 'prognostic\_fields' to save GPU memory

### Removed
- climetlab


## [0.1.10] Fix missing constants

### Added
- (GH) Added downstream-ci, reathedocs update check and label public pr workflows

### Changed
- Fix missing constant_fields property to query constants in the checkpoint

## [0.1.9] Patch, Move output finalise to ai-models

### Removed
- output finalise in the plugin

## [0.1.8] Patch, Support for output finalise in the plugin

### Added
- Support for output finalise in the plugin

## [0.1.7] Patch, graph utility

### Added
- graph utility

### Changed
- updated dependencies

## [0.1.6] Patch, update dependencies

### Changed
- updated dependencies

## [0.1.5] Patch, inspect cli tool

### Added
- tests
- inspect cli tool

## [0.1.4] Patch, autocast option

### Added
- add autocast option

## [0.1.3] Patch, support ai-models

### Added
- ai-models and AIModelPlugin

## [0.1.2] Patch

### Added
- dependency group all

## [0.1.0] Initial Release

### Added
Initial Implementation of anemoi-inference

## Git Diffs:
[Unreleased]: https://github.com/ecmwf/anemoi-inference/compare/0.1.10...HEAD
[0.1.10]: https://github.com/ecmwf/anemoi-inference/compare/0.1.9...0.1.10
[0.1.9]: https://github.com/ecmwf/anemoi-inference/compare/0.1.8...0.1.9
[0.1.8]: https://github.com/ecmwf/anemoi-inference/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/ecmwf/anemoi-inference/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/ecmwf/anemoi-inference/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/ecmwf/anemoi-inference/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/ecmwf/anemoi-inference/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/ecmwf/anemoi-inference/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/ecmwf/anemoi-inference/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/ecmwf/anemoi-inference/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/ecmwf/anemoi-inference/releases/tag/0.1.0
