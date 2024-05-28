#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import json
import logging
import os
import subprocess
from tempfile import TemporaryDirectory

import yaml

from . import Command

LOG = logging.getLogger(__name__)


class Metadata(Command):
    """Edit, remove, dump or load metadata from a checkpoint file."""

    def add_arguments(self, command_parser):
        from anemoi.utils.checkpoints import DEFAULT_NAME

        command_parser.add_argument("path", help="Path to the checkpoint.")

        group = command_parser.add_mutually_exclusive_group(required=True)

        group.add_argument(
            "--dump",
            action="store_true",
            help=(
                "Extract the metadata from the checkpoint and print it to the standard output"
                " or the file specified by ``--output``, in JSON or YAML format."
            ),
        )
        group.add_argument(
            "--load",
            action="store_true",
            help=(
                "Set the metadata in the checkpoint from the content"
                " of a file specified by the ``--input`` argument."
            ),
        )

        group.add_argument(
            "--edit",
            action="store_true",
            help=(
                "Edit the metadata in place, using the specified editor."
                " See the ``--editor`` argument for more information."
            ),
        )

        group.add_argument(
            "--remove",
            action="store_true",
            help="Remove the metadata from the checkpoint.",
        )

        command_parser.add_argument(
            "--name",
            default=DEFAULT_NAME,
            help="Name of metadata record to be used with the actions above.",
        )

        command_parser.add_argument(
            "--input",
            help="The output file name to be used by the ``--load`` option.",
        )

        command_parser.add_argument(
            "--output",
            help="The output file name to be used by the ``--dump`` option.",
        )

        command_parser.add_argument(
            "--editor",
            help="Editor to use for the ``--edit`` option. Default to ``$EDITOR`` if defined, else ``vi``.",
            default=os.environ.get("EDITOR", "vi"),
        )

        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Use the JSON format with ``--dump`` and ``--edit``.",
        )

        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Use the YAML format with ``--dump`` and ``--edit``.",
        )

    def run(self, args):
        if args.edit:
            return self.edit(args)

        if args.remove:
            return self.remove(args)

        if args.dump:
            return self.dump(args)

        if args.load:
            return self.load(args)

    def edit(self, args):

        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import replace_metadata

        if args.json:
            ext = "json"
            dump = json.dump
            load = json.load
            kwargs = {"indent": 4, "sort_keys": True}
        else:
            ext = "yaml"
            dump = yaml.dump
            load = yaml.safe_load
            kwargs = {"default_flow_style": False}

        with TemporaryDirectory() as temp_dir:

            path = os.path.join(temp_dir, f"checkpoint.{ext}")
            metadata = load_metadata(args.path)

            with open(path, "w") as f:
                dump(metadata, f, **kwargs)

            subprocess.check_call([args.editor, path])

            with open(path) as f:
                edited = load(f)

            if edited != metadata:
                replace_metadata(args.path, edited)
            else:
                LOG.info("No changes made.")

    def remove(self, args):
        from anemoi.utils.checkpoints import remove_metadata

        remove_metadata(args.path, args.name)

    def dump(self, args):
        from anemoi.utils.checkpoints import load_metadata

        if args.output:
            file = open(args.output, "w")
        else:
            file = None

        metadata = load_metadata(args.path)

        if args.yaml:
            print(yaml.dump(metadata, indent=2, sort_keys=True), file=file)
            return

        if args.json or True:
            print(json.dumps(metadata, indent=4, sort_keys=True), file=file)
            return

    def load(self, args):
        from anemoi.utils.checkpoints import has_metadata
        from anemoi.utils.checkpoints import replace_metadata
        from anemoi.utils.checkpoints import save_metadata

        if args.input is None:
            raise ValueError("Please specify a value for --input")

        _, ext = os.path.splitext(args.input)
        if ext == ".json" or args.json:
            with open(args.input) as f:
                metadata = json.load(f)

        elif ext in (".yaml", ".yml") or args.yaml:
            with open(args.input) as f:
                metadata = yaml.safe_load(f)

        else:
            raise ValueError(f"Unknown file extension {ext}. Please specify --json or --yaml")

        if has_metadata(args.path, args.name):
            replace_metadata(args.path, metadata)
        else:
            save_metadata(args.path, metadata, args.name)


command = Metadata
