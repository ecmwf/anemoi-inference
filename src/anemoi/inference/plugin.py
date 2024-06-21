# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
import os

import tqdm
from ai_models.model import Model

from anemoi.inference.runner import AUTOCAST
from anemoi.inference.runner import DefaultRunner

LOG = logging.getLogger(__name__)


class AIModelPlugin(Model):

    expver = None

    def add_model_args(self, parser):
        """To be implemented in subclasses to add model-specific arguments to the parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            An instance of the parser to add arguments to.
        """
        pass

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument("--checkpoint", required=not hasattr(self, "download_files"))
        parser.add_argument(
            "--autocast",
            type=str,
            choices=sorted(AUTOCAST.keys()),
        )

        self.add_model_args(parser)

        args = parser.parse_args(args)

        for k, v in vars(args).items():
            setattr(self, k, v)

        if args.checkpoint:
            self.runner = DefaultRunner(args.checkpoint)
        else:
            self.runner = DefaultRunner(os.path.join(self.assets, self.download_files[0]))

        return parser

    def run(self):
        if self.deterministic:
            self.torch_deterministic_mode()

        self.runner.run(
            input_fields=self.all_fields,
            lead_time=self.lead_time,
            start_datetime=self.start_datetime,
            device=self.device,
            output_callback=self._output,
            autocast=self.autocast,
            progress_callback=tqdm.tqdm,
        )

    def _output(self, *args, **kwargs):
        if "step" in kwargs or "endStep" in kwargs:
            return self.write(*args, **kwargs)

        return self.write_input_fields(*args, **kwargs)

    # Below are methods forwarded to the checkpoint

    @property
    def param_sfc(self):
        return self.runner.checkpoint.param_sfc

    @property
    def param_level_pl(self):
        return self.runner.checkpoint.param_level_pl

    @property
    def param_level_ml(self):
        return self.runner.checkpoint.param_level_ml

    @property
    def grid(self):
        return self.runner.checkpoint.grid

    @property
    def area(self):
        return self.runner.checkpoint.area

    @property
    def lagged(self):
        return self.runner.lagged


model = AIModelPlugin
