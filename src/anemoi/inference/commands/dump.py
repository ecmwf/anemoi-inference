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

import yaml

from ..checkpoint import Checkpoint
from . import Command


def _dump(x, path):
    if isinstance(x, dict):
        for k, v in x.items():

            path.append(k)
            _dump(v, path)
            path.pop()
        return

    if isinstance(x, list):
        for i, v in enumerate(x):
            path.append(str(i))
            _dump(v, path)
            path.pop()
        return

    name = ".".join(path)
    print(f"{name}: {repr(x)}")


class DumpCmd(Command):
    """Dump the content of a checkpoint file as JSON or YAML."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("--json", action="store_true", help="Output in JSON format")
        command_parser.add_argument("--yaml", action="store_true", help="Output in YAML format")
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):

        c = Checkpoint(args.path)
        if args.json:
            print(json.dumps(c.to_dict(), indent=4, sort_keys=True))
            return

        if args.yaml:
            print(yaml.dump(c.to_dict(), indent=4, sort_keys=True))
            return

        _dump(c.to_dict(), [])


command = DumpCmd
