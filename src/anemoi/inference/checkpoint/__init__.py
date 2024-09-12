# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
import os
from functools import cached_property
from typing import Literal

from anemoi.utils.checkpoints import has_metadata
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.provenance import gather_provenance_info

from .metadata import Metadata

LOG = logging.getLogger(__name__)


class Checkpoint:
    def __init__(self, path):
        self.path = path
        self._metadata = None
        self._operational_config = None

    def __repr__(self):
        return self.path

    def __getattr__(self, name):
        if self._metadata is None:
            try:
                self._metadata = Metadata.from_metadata(load_metadata(self.path))
            except ValueError:
                if has_metadata(self.path):
                    raise
                self._metadata = Metadata.from_metadata(None)

        return getattr(self._metadata, name)

    def _checkpoint_metadata(self, name):
        return load_metadata(self.path, name)

    @cached_property
    def operational_config(self):
        try:
            result = load_metadata(self.path, "operational-config.json")
            LOG.info(f"Using operational configuration from checkpoint {self.path}")
            return result
        except ValueError:
            pass

        # Check for the operational-config.json file in the model directory
        path = os.path.join(os.path.dirname(self.path), "operational-config.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                result = json.load(f)
                LOG.info(f"Using operational configuration from {path}")
                return result

        LOG.warning("No operational configuration found. Using default configuration.")
        return {}

    def validate_environment(
        self,
        all_packages: bool = False,
        on_difference: Literal["warn", "error"] = "warn",
    ) -> int:
        """
        Validate environment of the checkpoint against the current environment.

        Parameters
        ----------
        all_packages : bool, optional
            Check all packages in environment or just `anemoi`'s, by default False
        on_difference : Literal['warn', 'error'], optional
            What to do on difference, by default "warn"

        Returns
        -------
        int
            0 if environment is valid, 1 otherwise

        Raises
        ------
        RuntimeError
            If found difference and `on_difference` is 'error'
        ValueError
            If `on_difference` is not 'warn' or 'error'
        """
        train_environment = self.provenance_training
        inference_environment = gather_provenance_info(full=False)

        invalid_messages = {
            "python": [],
            "missing": [],
            "mismatch": [],
            "uncommitted": [],
        }

        if train_environment["python"] != inference_environment["python"]:
            invalid_messages["python"].append(
                f"Python version mismatch: {train_environment['python']} != {inference_environment['python']}"
            )

        for module in train_environment["module_versions"].keys():
            if not all_packages and "anemoi" not in module:
                continue

            if module not in inference_environment["module_versions"]:
                invalid_messages["missing"].append(f"Missing module in inference environment: {module}")
            elif train_environment["module_versions"][module] != inference_environment["module_versions"][module]:
                invalid_messages["mismatch"].append(
                    f"Version mismatch for module {module}: {train_environment['module_versions'][module]} != {inference_environment['module_versions'][module]}"
                )

        for git_record in train_environment["git_versions"].keys():
            file_record = train_environment["git_versions"][git_record]["git"]
            if file_record["modified_files"] == 0 and file_record["untracked_files"] == 0:
                continue

            if git_record not in inference_environment["git_versions"]:
                invalid_messages["uncommitted"].append(
                    f"Training environment contained uncommitted change missing in inference environment: {git_record}"
                )
            elif (
                train_environment["git_versions"][git_record]["sha1"]
                != inference_environment["git_versions"][git_record]["sha1"]
            ):
                invalid_messages["uncommitted"].append(
                    f"sha1 mismatch for git record between training and inference {git_record}: {train_environment['git_versions'][git_record]} != {inference_environment['git_versions'][git_record]}"
                )

        for git_record in inference_environment["git_versions"].keys():
            file_record = inference_environment["git_versions"][git_record]["git"]
            if file_record["modified_files"] == 0 and file_record["untracked_files"] == 0:
                continue

            if git_record not in train_environment["git_versions"]:
                invalid_messages["uncommitted"].append(
                    f"Inference environment contains uncommited changes missing in training: {git_record}"
                )

        if len(invalid_messages) > 0:
            text = "Environment validation failed. The following issues were found:\n" + "\n".join(
                [f"  {key}:\n    " + "\n    ".join(value) for key, value in invalid_messages.items()]
            )
            if on_difference == "warn":
                LOG.warning(text)
            elif on_difference == "error":
                raise RuntimeError(text)
            else:
                raise ValueError(f"Invalid value for `on_difference`: {on_difference}")
            return 1

        LOG.info(f"Environment validation passed")
        return 0
