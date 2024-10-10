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


class RequestCmd(Command):
    """Inspect the contents of a checkpoint file."""

    need_logging = False

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):

        c = Checkpoint(args.path)

        for r in c.retrieve_request():
            print(r)


command = RequestCmd
