# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.4.10](https://github.com/ecmwf/anemoi-inference/compare/0.4.9...0.4.10) (2025-02-25)


### Features

* Add truth output ([#144](https://github.com/ecmwf/anemoi-inference/issues/144)) ([cfefc21](https://github.com/ecmwf/anemoi-inference/commit/cfefc21743802af3ca19a62cbbeb9e501f49cd5a))
* **output:** allow selecting variables for output ([#118](https://github.com/ecmwf/anemoi-inference/issues/118)) ([3c833e1](https://github.com/ecmwf/anemoi-inference/commit/3c833e19f23eac584c59845070d49778fdf65b9a))
* parallel inference without slurm ([#121](https://github.com/ecmwf/anemoi-inference/issues/121)) ([90d7911](https://github.com/ecmwf/anemoi-inference/commit/90d79111a71963c560e026b67e9625ef195d2fbc))
* **retrieve:** Create runner from config ([#146](https://github.com/ecmwf/anemoi-inference/issues/146)) ([e7283b9](https://github.com/ecmwf/anemoi-inference/commit/e7283b9cf7d4622cabd69a297b4543525cfd479c))


### Bug Fixes

* issue [#127](https://github.com/ecmwf/anemoi-inference/issues/127), command "requests" broken ([#141](https://github.com/ecmwf/anemoi-inference/issues/141)) ([abfb633](https://github.com/ecmwf/anemoi-inference/commit/abfb63377f13cf4afc5bb6dfc8292a0d81afc444))
* prepml staging ([#150](https://github.com/ecmwf/anemoi-inference/issues/150)) ([384c5ee](https://github.com/ecmwf/anemoi-inference/commit/384c5ee59d19f631d8bc621256e86359b8f92aeb))
* Update output printer to avoid range(a, b, 0) ([#138](https://github.com/ecmwf/anemoi-inference/issues/138)) ([7cb2f0d](https://github.com/ecmwf/anemoi-inference/commit/7cb2f0d8e983350448a1c94e11625c740623ae5b))

## 0.4.9 (2025-02-13)

<!-- Release notes generated using configuration in .github/release.yml at main -->



**Full Changelog**: https://github.com/ecmwf/anemoi-inference/compare/0.4.8...0.4.9

## 0.4.8 (2025-02-11)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Exciting New Features 🎉
* feat(config): Add `accumulate_from_start_of_forecast` post-processor by @gmertes in https://github.com/ecmwf/anemoi-inference/pull/133
### Other Changes 🔗
* refactor: Rename `accumulations` -> `accumulate_from_start_of_forecast` by @gmertes in https://github.com/ecmwf/anemoi-inference/pull/135
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-inference/pull/136


**Full Changelog**: https://github.com/ecmwf/anemoi-inference/compare/0.4.7...0.4.8

## 0.4.7 (2025-02-10)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Exciting New Features 🎉
* feat(retrieve): Add `--use-scda` flag by @gmertes in https://github.com/ecmwf/anemoi-inference/pull/132
### Other Changes 🔗
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-inference/pull/126
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-inference/pull/130


**Full Changelog**: https://github.com/ecmwf/anemoi-inference/compare/0.4.6...0.4.7

## 0.4.6 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Other Changes 🔗
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-inference/pull/113

## New Contributors
* @DeployDuck made their first contribution in https://github.com/ecmwf/anemoi-inference/pull/113

**Full Changelog**: https://github.com/ecmwf/anemoi-inference/compare/0.4.5...0.4.6

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
- Add support for `output_frequency` to write less output
- Added ability to run inference over multiple GPUs [#55](https://github.com/ecmwf/anemoi-inference/pull/55)

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
