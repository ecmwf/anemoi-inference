# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from argparse import ArgumentParser
from argparse import Namespace

from anemoi.utils.grib import shortname_to_paramid

from ..checkpoint import Checkpoint
from . import Command
from .retrieve import checkpoint_to_requests


class RequestCmd(Command):
    """MARS request utility."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.description = self.__doc__
        command_parser.add_argument("--mars", action="store_true", help="Print the MARS request.")
        command_parser.add_argument("--use-grib-paramid", action="store_true", help="Use paramId instead of param.")
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args: Namespace) -> None:
        """Run the request command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        c = Checkpoint(args.path)
        for r in checkpoint_to_requests(c, date=-1, use_grib_paramid=args.use_grib_paramid):
            if args.mars:
                req = ["retrieve,target=data"]
                for k, v in r.items():

                    if args.use_grib_paramid and k == "param":
                        if not isinstance(v, (list, tuple)):
                            v = [v]
                        v = [shortname_to_paramid(x) for x in v]

                    if isinstance(v, (list, tuple)):
                        v = "/".join([str(x) for x in v])
                    req.append(f"{k}={v}")
                r = ",".join(req)
            print(r)


command = RequestCmd
