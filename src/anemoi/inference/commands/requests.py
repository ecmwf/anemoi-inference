#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging

from ..checkpoint import Checkpoint
from . import Command

LOG = logging.getLogger(__name__)


class RequestsCmd(Command):
    """Patch a checkpoint file."""

    need_logging = False
    _cache = {}

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args):
        c = Checkpoint(args.path)

        print(c.request())


command = RequestsCmd
