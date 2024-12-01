# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import yaml

from ..tasks import create_task
from ..transports import create_transport
from . import Command

LOG = logging.getLogger(__name__)


class Coupling:
    """_summary_"""

    def __init__(self, source, sidx, target, tidx):
        self.source = source
        self.sidx = sidx
        self.target = target
        self.tidx = tidx

    def __str__(self):
        return f"{self.source}:{self.sidx}->{self.target}:{self.tidx}"


class CouplingSend(Coupling):
    """_summary_"""

    def apply(self, task, transport, tensor, tag):
        # print(f"{RANK}: sending {self} {tag}")
        # COMM.Send(tensor[self.sidx], dest=self.target.rank, tag=tag)
        # print(f"{RANK}: sent from {self}  {tag}")
        transport.send(task, tensor[self.sidx], self.target, tag)


class CouplingRecv(Coupling):
    """_summary_"""

    def apply(self, task, transport, tensor, tag):
        # print(f"{RANK}: receiving {self}  {tag}")
        # COMM.Recv(tensor[self.tidx], source=self.source.rank, tag=tag)
        # print(f"{RANK}: received {self}  {tag}")
        transport.receive(task, tensor[self.tidx], self.source, tag)


class CoupleCmd(Command):
    """Inspect the contents of a checkpoint file."""

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to config file.")
        command_parser.add_argument("overrides", nargs="*", help="Overrides.")

    def run(self, args):

        config = yaml.safe_load(open(args.config))

        tasks = {name: create_task(name, action) for name, action in config["tasks"].items()}
        for task in tasks.values():
            LOG.info("Task: %s", task)

        couplings = []
        for coupling in config["couplings"]:
            source, target = coupling.split("->")
            source, sidx = source.strip().split(":")
            target, tidx = target.strip().split(":")

            couplings.append(
                CouplingSend(
                    tasks[source],
                    int(sidx),
                    tasks[target],
                    int(tidx),
                )
            )

            couplings.append(
                CouplingRecv(
                    tasks[source],
                    int(sidx),
                    tasks[target],
                    int(tidx),
                )
            )

        transport = create_transport(config["transport"], couplings)
        LOG.info("Transport: %s", transport)

        transport.start(tasks)
        transport.wait()


command = CoupleCmd
