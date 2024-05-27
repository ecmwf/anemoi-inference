#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from ..checkpoint import Checkpoint
from . import Command


def visit(x, path, name, value):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == name:
                print(".".join(path), k, v)

            if v == value:
                print(".".join(path), k, v)

            path.append(k)
            visit(v, path, name, value)
            path.pop()

    if isinstance(x, list):
        for i, v in enumerate(x):
            path.append(str(i))
            visit(v, path, name, value)
            path.pop()


class EditCmd(Command):

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument("--name", help="Search for a specific name.")
        command_parser.add_argument("--value", help="Search for a specific value.")

    def run(self, args):

        checkpoint = Checkpoint(args.path)

        visit(
            checkpoint,
            [],
            args.name if args.name is not None else object(),
            args.value if args.value is not None else object(),
        )


command = EditCmd
