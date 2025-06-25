# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.5.4](https://github.com/ecmwf/anemoi-inference/compare/0.5.3...0.5.4) (2025-05-08)


### Features

* Support vscode metadata editor ([#228](https://github.com/ecmwf/anemoi-inference/issues/228)) ([e3dc9bb](https://github.com/ecmwf/anemoi-inference/commit/e3dc9bb520bd3ebec6fa7c70ccdbdfc25a62e2a4))


### Bug Fixes

* boundary forcings inputs ([#226](https://github.com/ecmwf/anemoi-inference/issues/226)) ([02eaa6b](https://github.com/ecmwf/anemoi-inference/commit/02eaa6b4dd2227a049860aa16f89bcb0762a63a3))
* support for netcdf missing values ([#214](https://github.com/ecmwf/anemoi-inference/issues/214)) ([4e78971](https://github.com/ecmwf/anemoi-inference/commit/4e78971e8affcb345c2ec7cd880e2f4d0d5d673a))

## [0.5.3](https://github.com/ecmwf/anemoi-inference/compare/0.5.2...0.5.3) (2025-04-30)


### Features

* **input:** add FDB input class ([#190](https://github.com/ecmwf/anemoi-inference/issues/190)) ([ca6d37f](https://github.com/ecmwf/anemoi-inference/commit/ca6d37f9d15a608b2b63b8ec05544dd051d4d7d2))
* more flexible supporting array path for extract lam ([#219](https://github.com/ecmwf/anemoi-inference/issues/219)) ([edd176f](https://github.com/ecmwf/anemoi-inference/commit/edd176fd75d0700b3a3a5faf247fca23b03ced60))


### Bug Fixes

* Allow for downstream tests to provide checkpoint mocks ([#211](https://github.com/ecmwf/anemoi-inference/issues/211)) ([fbc3399](https://github.com/ecmwf/anemoi-inference/commit/fbc33992844cdda01d91396427ec5b8d220fe28f))
* better error messages ([#213](https://github.com/ecmwf/anemoi-inference/issues/213)) ([746ecfe](https://github.com/ecmwf/anemoi-inference/commit/746ecfe0cb7c8045cea5e7dcaefd68f7943f0c7d))
* extract-lam ([#217](https://github.com/ecmwf/anemoi-inference/issues/217)) ([db81c4c](https://github.com/ecmwf/anemoi-inference/commit/db81c4c0f92a3d3428d4b68f9dce6c4bd5861aaf))
* Inference with crps requires additional arguments in predict_step ([#212](https://github.com/ecmwf/anemoi-inference/issues/212)) ([7917d53](https://github.com/ecmwf/anemoi-inference/commit/7917d5368ba8518bab0f8848c4f4981250af9ab2))

## [0.5.2](https://github.com/ecmwf/anemoi-inference/compare/0.5.1...0.5.2) (2025-04-25)


### Features

* Add version validation if model fails to load ([#194](https://github.com/ecmwf/anemoi-inference/issues/194)) ([bfd890f](https://github.com/ecmwf/anemoi-inference/commit/bfd890f3f656d8e91cacbcb2cd4f5e7c93853879))
* Allow empty config path to run from overrides ([#206](https://github.com/ecmwf/anemoi-inference/issues/206)) ([a5177d3](https://github.com/ecmwf/anemoi-inference/commit/a5177d3fce8abb6d93ad0c9e1b2b4066c8b3fc06))
* Allow hydra-like override of config list elements ([#201](https://github.com/ecmwf/anemoi-inference/issues/201)) ([ff0c607](https://github.com/ecmwf/anemoi-inference/commit/ff0c6077ae49ea4e873bba9575565e1b132cb912))


### Bug Fixes

* Hindcast encoding ([217d81e](https://github.com/ecmwf/anemoi-inference/commit/217d81e966ecba830ded4d9c6f6634e965b824ed))
* Improve Testing Decorators ([#209](https://github.com/ecmwf/anemoi-inference/issues/209)) ([03a7f65](https://github.com/ecmwf/anemoi-inference/commit/03a7f65dc0f983d9194c1471ec10c1e3bb20adc2))


### Documentation

* Fix cutout example ([ff9715c](https://github.com/ecmwf/anemoi-inference/commit/ff9715cec988c011fe58d2c60bd630bf4d746822))

## [0.5.1](https://github.com/ecmwf/anemoi-inference/compare/0.5.0...0.5.1) (2025-04-09)


### Features

* plugin support ([#187](https://github.com/ecmwf/anemoi-inference/issues/187)) ([531e2ad](https://github.com/ecmwf/anemoi-inference/commit/531e2ad1c11b946f8e37448fea9449eb5f798cbd))


### Bug Fixes

* Disable coupled test to fix runners ([8190dc0](https://github.com/ecmwf/anemoi-inference/commit/8190dc0d41d4bedc89f0ebfce82f0237bf96f720))
* naming of pre/post processed fields ([#198](https://github.com/ecmwf/anemoi-inference/issues/198)) ([a082af4](https://github.com/ecmwf/anemoi-inference/commit/a082af43ee214ad956315ef357f5d8d68842a662))
* **netcdf:** Fix context time attributes ([#204](https://github.com/ecmwf/anemoi-inference/issues/204)) ([d6fd430](https://github.com/ecmwf/anemoi-inference/commit/d6fd4304779f074fe2cc3f00d1b777ac7c6f02d8))
* **retrieve:** Patch scda regardless of start date ([b67562a](https://github.com/ecmwf/anemoi-inference/commit/b67562a9325eca223e35818c8b15d6edd87ca465))
* Write correct values at step 0 ([#186](https://github.com/ecmwf/anemoi-inference/issues/186)) ([7923860](https://github.com/ecmwf/anemoi-inference/commit/7923860abd237b463d035a5522975547b54f619d))


### Documentation

* align docs with template ([#195](https://github.com/ecmwf/anemoi-inference/issues/195)) ([f444c52](https://github.com/ecmwf/anemoi-inference/commit/f444c5282f9bdb36bceb9be3316eb49356d8c24e))

## [0.5.0](https://github.com/ecmwf/anemoi-inference/compare/0.4.11...0.5.0) (2025-03-24)


### ⚠ BREAKING CHANGES

* Implement coupling of two or more models ([#76](https://github.com/ecmwf/anemoi-inference/issues/76))

### Features

* Implement coupling of two or more models ([#76](https://github.com/ecmwf/anemoi-inference/issues/76)) ([94e6a7c](https://github.com/ecmwf/anemoi-inference/commit/94e6a7cbc796294d76a05efa1d9281ede0ba0501))
* support for cutout input ([#169](https://github.com/ecmwf/anemoi-inference/issues/169)) ([7f38095](https://github.com/ecmwf/anemoi-inference/commit/7f3809501e8ae8b603fa9dd94d5d224898b7f920))


### Bug Fixes

* Add check for date is None with earthkit-data GRIB input ([#184](https://github.com/ecmwf/anemoi-inference/issues/184)) ([6bfe48f](https://github.com/ecmwf/anemoi-inference/commit/6bfe48fa890311f64edf521be66ac0818b7838a5))
* **grib:** Remove duplicate grib key logic ([#177](https://github.com/ecmwf/anemoi-inference/issues/177)) ([3156e0f](https://github.com/ecmwf/anemoi-inference/commit/3156e0fc9ac38bd2181d12c70f0c6578e632ff4c))
* ParallelRunner without GPU ([#174](https://github.com/ecmwf/anemoi-inference/issues/174)) ([9313155](https://github.com/ecmwf/anemoi-inference/commit/9313155e7bc19b0fce51f6330c98761343ccfb54))
* param id handling at archive JSON output ([#172](https://github.com/ecmwf/anemoi-inference/issues/172)) ([ca3c500](https://github.com/ecmwf/anemoi-inference/commit/ca3c500df8d75c0706e525d5f98173d82af344fc))
* pyproject dependency on ai-models ([#192](https://github.com/ecmwf/anemoi-inference/issues/192)) ([f59d926](https://github.com/ecmwf/anemoi-inference/commit/f59d926f7e6fbe932295178296f5bbefb94bbe61))
* Restore default accumulation behaviour ([#131](https://github.com/ecmwf/anemoi-inference/issues/131)) ([c645fb2](https://github.com/ecmwf/anemoi-inference/commit/c645fb219ed40ea60389affafe0f4f6c3b799ee9))
* unused parameter ([#175](https://github.com/ecmwf/anemoi-inference/issues/175)) ([3d7c507](https://github.com/ecmwf/anemoi-inference/commit/3d7c507b2c0927be21e08819776b4e9b456e39ef))
* use processes in coupling tests ([#193](https://github.com/ecmwf/anemoi-inference/issues/193)) ([30b65df](https://github.com/ecmwf/anemoi-inference/commit/30b65df112c072a887fba02daf0994bcf5b5bc25))


### Documentation

* Docathon ([#183](https://github.com/ecmwf/anemoi-inference/issues/183)) ([cbecfc9](https://github.com/ecmwf/anemoi-inference/commit/cbecfc9198bf462b65402b266d5b3c6c469eb755))

## [0.4.11](https://github.com/ecmwf/anemoi-inference/compare/0.4.10...0.4.11) (2025-03-07)


### Features

* Add metadata --get functionality ([#154](https://github.com/ecmwf/anemoi-inference/issues/154)) ([6821905](https://github.com/ecmwf/anemoi-inference/commit/68219051664a1be955b7ae800e1052eb69feb79c))
* add time and memory profiler to runner ([#143](https://github.com/ecmwf/anemoi-inference/issues/143)) ([a0c4bb4](https://github.com/ecmwf/anemoi-inference/commit/a0c4bb4e1629df3e475772839fe71be94e6b5b9f))
* adding post-processors ([#116](https://github.com/ecmwf/anemoi-inference/issues/116)) ([9e240fa](https://github.com/ecmwf/anemoi-inference/commit/9e240fa536b43c2d405c934965bd6bd2463fd8f1))
* **runner:** Forecast loop step generator ([#168](https://github.com/ecmwf/anemoi-inference/issues/168)) ([cc7fed5](https://github.com/ecmwf/anemoi-inference/commit/cc7fed54aa5f1923da7ef08649b102e2e690a3bf))


### Bug Fixes

* fix pyproject.toml ([#163](https://github.com/ecmwf/anemoi-inference/issues/163)) ([104016b](https://github.com/ecmwf/anemoi-inference/commit/104016b93294fb0f767add1b218929734d740383))
* Post processors ([#164](https://github.com/ecmwf/anemoi-inference/issues/164)) ([c5bd6a9](https://github.com/ecmwf/anemoi-inference/commit/c5bd6a9559c1306bfbbf7aee10a1e681211b1f73))
* Remove `hdate` from variable keys ([#165](https://github.com/ecmwf/anemoi-inference/issues/165)) ([c024ddf](https://github.com/ecmwf/anemoi-inference/commit/c024ddf0dcc5b18acf6e9949b2444b559b00d423))
* **retrieve:** Missing arguments ([8d7b174](https://github.com/ecmwf/anemoi-inference/commit/8d7b174bb45259ed2dda48b11a748323f6ca8afe))
* **retrieve:** Set target from config before extras ([8b22ea7](https://github.com/ecmwf/anemoi-inference/commit/8b22ea73755c05a61d48aa2c7506565df4864f45))
* Rework Truth Output ([#159](https://github.com/ecmwf/anemoi-inference/issues/159)) ([7601c2e](https://github.com/ecmwf/anemoi-inference/commit/7601c2e1d39a42cf5b9d8dee9b7a1a0345fe478a))
* **run:** Add processors to `_run` wrapper ([e4261e7](https://github.com/ecmwf/anemoi-inference/commit/e4261e7346cd95d63036fdd827f094701187c990))
* Update Profiler ([#160](https://github.com/ecmwf/anemoi-inference/issues/160)) ([6cfa021](https://github.com/ecmwf/anemoi-inference/commit/6cfa021ec8cdfc9b18a5bc51a7937759e4c73e28))


### Documentation

* Add validation info ([#151](https://github.com/ecmwf/anemoi-inference/issues/151)) ([f132803](https://github.com/ecmwf/anemoi-inference/commit/f132803fd368ee1148b6ba207570b796fe79e285))
* use new logo ([#142](https://github.com/ecmwf/anemoi-inference/issues/142)) ([9a3a2fb](https://github.com/ecmwf/anemoi-inference/commit/9a3a2fba058422c075d86f9f71cc91a1a68617b6))

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
