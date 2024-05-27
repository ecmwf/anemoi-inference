#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import os
import subprocess
from tempfile import TemporaryDirectory

import yaml

from . import Command


class EditCmd(Command):

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument("--editor", help="Editor to use.", default=os.environ.get("EDITOR", "vi"))

    def run(self, args):

        from anemoi.utils.checkpoints import DEFAULT_NAME
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import metadata_files
        from anemoi.utils.checkpoints import replace_metadata

        OLD_NAME = "ai-models.json"

        names = metadata_files(args.path)
        name = OLD_NAME if OLD_NAME in names else DEFAULT_NAME

        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "checkpoint.yaml")
            with open(path, "w") as f:
                yaml.dump(load_metadata(args.path, name), f)

            subprocess.check_call([args.editor, path])

            with open(path) as f:
                replace_metadata(args.path, yaml.safe_load(f), OLD_NAME)

            # checkpoint.pack(temp_dir, args.path)


command = EditCmd
