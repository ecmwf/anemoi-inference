# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import sys
import warnings
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import rich
import yaml

from ..checkpoint import Checkpoint
from . import Command

CORE = ("models", "training", "graphs")


SKIP = {"anemoi-training"}

SKIP |= set(sys.stdlib_module_names) | set(sys.builtin_module_names)

PACKAGES = {
    "sklearn": "scikit-learn",
    "attr": "attrs",
    "google-protobuf": "protobuf",
}


class InspectCmd(Command):
    """Inspect the contents of a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint.")
        group = command_parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--variables", action="store_true", help="List the training variables and their categories.")
        group.add_argument(
            "--requirements",
            action="store_true",
            help="Print a Python's requirements.txt based on the versions of the packages used during training.",
        )
        group.add_argument(
            "--datasets", action="store_true", help="Print the arguments passed to anemoi-dataset during training."
        )

        group.add_argument("--dump", action="store_true", help="Dump information from the checkpoint.")

        command_parser.add_argument("--json", action="store_true", help="Output in JSON format (with dump option)")
        command_parser.add_argument("--validate", action="store_true", help="Validate the environment.")

    def run(self, args: Namespace) -> None:
        """Run the inspect command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """

        c = Checkpoint(args.path)

        if args.validate:
            c.validate_environment()
            return

        if args.variables:
            self.variables(c, args)
            return

        if args.requirements:
            self.requirements(c, args)
            return

        if args.datasets:
            self.datasets(c, args)
            return

        if args.dump:
            self.dump(c, args)
            return

    def dump(self, c: Checkpoint, args: Namespace) -> None:

        if args.json:
            # turn off all other logging so json output can be piped cleanly
            logging.disable(logging.INFO)
            warnings.filterwarnings("ignore")

        def _(f: Callable[[], Any]) -> Any:
            """Wrapper function to handle exceptions.

            Parameters
            ----------
            f : Callable
                The function to be called.

            Returns
            -------
            Any
                The result of the function call or the exception message.
            """
            try:
                return f()
            except Exception as e:
                return str(e)

        data = {}
        for name in sorted(dir(c)):

            if name.startswith("_"):
                continue

            data[name] = _(lambda: getattr(c, name))

        if args.json:
            print(json.dumps(data, indent=None, default=str))
        else:
            for key, value in data.items():
                print(key, ":")
                print("  ", json.dumps(value, indent=4, default=str))
                print()

    def variables(self, c: Checkpoint, args: Namespace) -> None:
        """Print the variable categories in the checkpoint.

        Parameters
        ----------
        c : Checkpoint
            The checkpoint object.
        args : Namespace
            The command-line arguments.
        """
        c.print_variable_categories(print=rich.print)

    def indices(self, c: Checkpoint, args: Namespace) -> None:
        """Print the variable indices in the checkpoint.

        Parameters
        ----------
        c : Checkpoint
            The checkpoint object.
        args : Namespace
            The command-line arguments.
        """
        c.print_indices(print=rich.print)

    def datasets(self, c: Checkpoint, args: Namespace) -> None:
        """Print the dataset arguments and keyword arguments for opening datasets.

        Parameters
        ----------
        c : Checkpoint
            The checkpoint object.
        args : Namespace
            The command-line arguments.
        """
        open_dataset_args, open_dataset_kwargs = c.open_dataset_args_kwargs(use_original_paths=False)

        print("Open dataset arguments:")

        if open_dataset_args:
            print()
            print(yaml.dump(open_dataset_args, indent=4, default_flow_style=False))

        if open_dataset_kwargs:
            print()
            print(yaml.dump(open_dataset_kwargs, indent=4, default_flow_style=False))

    def requirements(self, c: Checkpoint, args: Namespace) -> None:
        """Print the requirements for the checkpoint, including PyPI and Git dependencies.

        Parameters
        ----------
        c : Checkpoint
            The checkpoint object.
        args : Namespace
            The command-line arguments.
        """
        c = Checkpoint(args.path)
        r = c.provenance_training()

        f = sys.stdout

        if args.dump:
            print(json.dumps(r, indent=2, sort_keys=True), file=f)
            return

        print("# This file is automatically generated from a checkpoint.", file=f)
        print("# Checkpoint:", args.path, file=f)
        print("# Python:", r.get("python"), file=f)
        print(file=f)

        distribution_names = r.get("distribution_names", {})
        distribution_names.update(PACKAGES)

        pypi_requirements = {}
        git_requirements = {}

        for k, v in r.get("module_versions", {}).items():
            if k.startswith("_"):
                continue
            if v[0].isdigit():
                v = [x for x in v.split(".") if x.isdigit()]
                v = ".".join(v)
                k = k.replace(".", "-")
                pypi_requirements[distribution_names.get(k, k)] = v

        for k, v in r.get("git_versions", {}).items():
            if not k.startswith("anemoi."):
                continue

            sha1 = v.get("git", {}).get("sha1")

            if not sha1:
                continue

            what = k.split(".")[-1]
            if what in CORE:
                url = f"git+https://github.com/ecmwf/anemoi-core@{sha1}#subdirectory={what}"
            else:
                url = f"git+https://github.com/ecmwf/anemoi-{what}@{sha1}"

            k = k.replace(".", "-")
            git_requirements[k] = url

        if git_requirements:
            print(file=f)
            print("# Git requirements:", file=f)
            print(file=f)

        for k, v in sorted(git_requirements.items()):
            if k in SKIP:
                continue

            version = pypi_requirements.pop(k, None)
            if version:
                print(f"# {k}=={version}", file=f)
            print(v, file=f)

        if pypi_requirements:
            print(file=f)
            print("# PyPI requirements:", file=f)
            print(file=f)

        for k, v in sorted(pypi_requirements.items()):
            if k in SKIP:
                continue

            print(f"{k}=={v}", file=f)


command = InspectCmd
