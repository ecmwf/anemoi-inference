# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

from anemoi.utils.provenance import gather_provenance_info
from packaging.version import Version

if TYPE_CHECKING:
    from anemoi.inference.metadata import Metadata


# Complete package name to be exempt
EXEMPT_PACKAGES = [
    "anemoi.training",
    "anemoi.inference",
    "hydra",
    "hydra_plugins",
    "hydra_plugins.anemoi_searchpath",
    "lightning",
    "pytorch_lightning",
    "lightning_fabric",
    "lightning_utilities",
]

# Entire namespaces to be exempt
EXEMPT_NAMESPACES = [
    "hydra_plugins",
]

LOG = logging.getLogger(__name__)


@overload
def validate_environment(
    metadata: "Metadata",
    *,
    all_packages: bool = False,
    on_difference: Literal["warn", "error", "ignore"] = "warn",
    exempt_packages: list[str] | None = None,
) -> bool: ...


@overload
def validate_environment(
    metadata: "Metadata",
    *,
    all_packages: bool = False,
    on_difference: Literal["return"] = "return",
    exempt_packages: list[str] | None = None,
) -> str: ...


def validate_environment(
    metadata: "Metadata",
    *,
    all_packages: bool = False,
    on_difference: Literal["warn", "error", "ignore", "return"] = "warn",
    exempt_packages: list[str] | None = None,
) -> bool | str:
    """Validate environment of the checkpoint against the current environment.

    Parameters
    ----------
    metadata : Metadata
        Metadata object of the checkpoint, to validate against
    all_packages : bool, optional
        Check all packages in environment or just `anemoi`'s, by default False
    on_difference : Literal['warn', 'error', 'ignore'], optional
        What to do on difference, by default "warn"
    exempt_packages : List[str], optional
        List of packages to exempt from the check, by default EXEMPT_PACKAGES

    Returns
    -------
    Union[bool, str]
        boolean if `on_difference` is not 'return', otherwise formatted text of the differences
        True if environment is valid, False otherwise

    Raises
    ------
    RuntimeError
        If found difference and `on_difference` is 'error'
    ValueError
        If `on_difference` is not 'warn' or 'error'
    """
    train_environment = metadata.provenance_training()
    inference_environment = gather_provenance_info(full=False)

    # Override module information with more complete inference environment capture
    import importlib.metadata as imp_metadata

    module_versions = {
        distribution.metadata.get("Name", "").replace("-", "_"): distribution.metadata["Version"]
        for distribution in imp_metadata.distributions()
    }

    inference_environment["module_versions"] = module_versions

    exempt_packages = exempt_packages or []
    exempt_packages.extend(EXEMPT_PACKAGES)

    invalid_messages: dict[str, list[str]] = {
        "python": [],
        "missing": [],
        "mismatch": [],
        "critical mismatch": [],
        "uncommitted": [],
    }

    if train_environment["python"] != inference_environment["python"]:
        invalid_messages["python"].append(
            f"Python version mismatch: {train_environment['python']} != {inference_environment['python']}"
        )

    for module in train_environment["module_versions"].keys():
        inference_module_name = module  # Due to package name differences between retrieval methods this may change

        train_module_version_str = train_environment["module_versions"][module]
        if not all_packages and "anemoi" not in module:
            continue
        elif module in exempt_packages or module.split(".")[0] in EXEMPT_NAMESPACES:
            continue
        elif module.startswith("_"):
            continue
        elif module not in inference_environment["module_versions"]:
            if "." in module and module.replace(".", "_") in inference_environment["module_versions"]:
                inference_module_name = module.replace(".", "_")
            else:
                try:
                    import importlib

                    importlib.import_module(module)
                    continue
                except (ModuleNotFoundError, ImportError):
                    pass
                invalid_messages["missing"].append(
                    f"Missing module in inference environment: {module}=={train_module_version_str}"
                )
                continue

        train_environment_version = Version(train_environment["module_versions"][module])
        inference_environment_version = Version(inference_environment["module_versions"][inference_module_name])

        if train_environment_version < inference_environment_version:
            invalid_messages["mismatch"].append(
                f"Version of module {module} was lower in training than in inference: {train_environment_version!s} <= {inference_environment_version!s}"
            )
        elif train_environment_version > inference_environment_version:
            invalid_messages["critical mismatch"].append(
                f"CRITICAL: Version of module {module} was greater in training than in inference: {train_environment_version!s} > {inference_environment_version!s}"
            )

    for git_record in train_environment["git_versions"].keys():
        file_record = train_environment["git_versions"][git_record]["git"]
        if file_record["modified_files"] == 0 and file_record["untracked_files"] == 0:
            continue

        if git_record in exempt_packages:
            continue

        if git_record not in inference_environment["git_versions"]:
            invalid_messages["uncommitted"].append(
                f"Training environment contained uncommitted change missing in inference environment: {git_record}"
            )
        elif (
            train_environment["git_versions"][git_record]["git"]["sha1"]
            != inference_environment["git_versions"][git_record]["git"]["sha1"]
        ):
            invalid_messages["uncommitted"].append(
                f"sha1 mismatch for git record between training and inference. {git_record} (training != inference): {train_environment['git_versions'][git_record]['git']} != {inference_environment['git_versions'][git_record]['git']}"
            )

    for git_record in inference_environment["git_versions"].keys():
        file_record = inference_environment["git_versions"][git_record]["git"]
        if file_record["modified_files"] == 0 and file_record["untracked_files"] == 0:
            continue

        if git_record in exempt_packages:
            continue

        if git_record not in train_environment["git_versions"]:
            invalid_messages["uncommitted"].append(
                f"Inference environment contains uncommited changes missing in training: {git_record}"
            )

    if len(invalid_messages) > 0:
        text = "Environment validation failed. The following issues were found:\n" + "\n".join(
            [f"  {key}:\n    " + "\n    ".join(value) for key, value in invalid_messages.items() if len(value) > 0]
        )
        if on_difference == "warn":
            LOG.warning(text)
        elif on_difference == "error":
            raise RuntimeError(text)
        elif on_difference == "ignore":
            pass
        elif on_difference == "return":
            return text
        else:
            raise ValueError(f"Invalid value for `on_difference`: {on_difference}")
        return False

    LOG.info("Environment validation passed")
    return True
