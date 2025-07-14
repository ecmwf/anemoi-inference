# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import sys
from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from anemoi.inference.types import BoolArray
from anemoi.inference.types import FloatArray

LOG = logging.getLogger(__name__)


class RolloutSource:
    """Represents a source of data that is a rollout."""

    trace_name = "rollout"


class UnknownSource:
    """Represents a source of data that is unknown."""

    trace_name = "?"


class UnchangedSource:
    """Represents a source of data that is unchanged."""

    trace_name = "unchanged"


class InputSource:
    """Represents a source of data that is an input."""

    def __init__(self, input: Any) -> None:
        self.input = input
        self.trace_name = input.trace_name


class Trace:
    """Implementation of a trace."""

    def __init__(self, path: Union[bool, str]) -> None:
        self.path = path
        self.file = sys.stdout if path is True else open(path, "w")

        self.sources: Dict[str, Any] = {}
        self.extra: Dict[str, Any] = {}

        print("-+" * 80, file=self.file)
        print("Trace:", datetime.datetime.now(), file=self.file)
        print("-+" * 80, file=self.file)
        print(file=self.file, flush=True)

    def write_input_tensor(
        self,
        date: datetime.datetime,
        fcstep: int,
        input_tensor: FloatArray,
        variable_to_input_tensor_index: Dict[str, int],
        timestep: datetime.timedelta,
    ) -> None:
        """Write the input tensor details to the trace file.

        Parameters
        ----------
        date : datetime.datetime
            The date associated with the input tensor.
        fcstep : int
            The step number.
        input_tensor : FloatArray
            The input tensor.
        variable_to_input_tensor_index : Dict[str, int]
            Mapping from variable names to input tensor indices.
        timestep : datetime.timedelta
            The timestep.
        """
        print(f"Input tensor to forecast date {date-timestep} => {date}", input_tensor.shape, file=self.file)
        names = {v: k for k, v in variable_to_input_tensor_index.items()}
        assert len(input_tensor.shape) == 4
        assert input_tensor.shape[0] == 1
        lines = []

        for s in range(input_tensor.shape[1]):
            for i in range(input_tensor.shape[-1]):
                values = input_tensor[0, s, :, i]
                if s == 0:
                    lines.append(
                        [names[i], np.nanmin(values), np.nanmax(values), np.nanmean(values), np.nanstd(values)]
                    )
                else:
                    constant = np.all(values == input_tensor[0, 0, :, i])
                    constant = "C" if constant else ""
                    lines[i].extend(
                        [np.nanmin(values), np.nanmax(values), np.nanmean(values), np.nanstd(values), constant]
                    )

        unknown = UnknownSource()
        for line in lines:
            line.append(self.sources.get(line[0], unknown).trace_name)

        lines.insert(0, ["--------", "---", "---", "----", "---", "---", "---", "----", "---", "-", "---"])
        lines.insert(0, ["Variable", "Min", "Max", "Mean", "Std", "Min", "Max", "Mean", "Std", "C", "Source"])
        self.table(lines)

        print(file=self.file, flush=True)

    def write_output_tensor(
        self,
        date: datetime.datetime,
        fcstep: int,
        output_tensor: FloatArray,
        output_tensor_index_to_variable: Dict[int, str],
        timestep: datetime.timedelta,
    ) -> None:
        """Write the output tensor details to the trace file.

        Parameters
        ----------
        date : datetime.datetime
            The date associated with the output tensor.
        fcstep : int
            The step number.
        output_tensor : FloatArray
            The output tensor.
        output_tensor_index_to_variable : Dict[int, str]
            Mapping from output tensor indices to variable names.
        timestep : datetime.timedelta
            The timestep.
        """
        print(f"Output tensor for {date}:", output_tensor.shape, file=self.file)
        assert len(output_tensor.shape) == 2
        names = output_tensor_index_to_variable
        lines = []
        for i in range(output_tensor.shape[-1]):
            values = output_tensor[:, i]
            lines.append([names[i], np.nanmin(values), np.nanmax(values), np.nanmean(values), np.nanstd(values)])

        lines.insert(0, ["--------", "---", "---", "----", "---"])
        lines.insert(0, ["Variable", "Min", "Max", "Mean", "Std"])
        self.table(lines)
        print(file=self.file, flush=True)

    def table(self, lines: list[list[Any]]) -> None:
        """Print a formatted table to the trace file.

        Parameters
        ----------
        lines : list[list[Any]]
            The table data to print.
        """
        print(file=self.file)

        def _(x: Any) -> str:
            if isinstance(x, float):
                return f"{x:g}"
            return str(x)

        lines = [[_(x) for x in line] for line in lines]

        lengths = [max(len(str(x)) for x in col) for col in zip(*lines)]
        for line in lines:
            print(" ".join(f"{x:{lengths[i]}}" for i, x in enumerate(line)), file=self.file)

    def from_source(self, name: str, source: Any, extra: Any = None) -> None:
        """Add a source to the trace.

        Parameters
        ----------
        name : str
            The name of the source.
        source : Any
            The source object.
        extra : Any, optional
            Additional information about the source.
        """
        if name in self.sources:
            old = self.sources[name]
            LOG.warning(
                f"Warning: source {name} already defined as '{old.trace_name}', changing to '{source.trace_name}'"
            )

        # assert name not in self.sources, name
        self.sources[name] = source
        self.extra[name] = extra

    def from_rollout(self, name: str) -> None:
        """Add a rollout source to the trace.

        Parameters
        ----------
        name : str
            The name of the rollout source.
        """
        self.from_source(name, RolloutSource())

    def from_input(self, name: str, input: Any) -> None:
        """Add an input source to the trace.

        Parameters
        ----------
        name : str
            The name of the input source.
        input : Any
            The input object.
        """
        self.from_source(name, InputSource(input))

    def reset_sources(self, reset: BoolArray, variable_to_input_tensor_index: Dict[str, int]) -> None:
        """Reset the sources in the trace.

        Parameters
        ----------
        reset : list[bool]
            List indicating which sources to reset.
        variable_to_input_tensor_index : Dict[str, int]
            Mapping from variable names to input tensor indices.
        """
        self.sources = {}
        self.extra = {}
        names = {v: k for k, v in variable_to_input_tensor_index.items()}
        for i, on in enumerate(reset):
            if on:
                self.sources[names[i]] = UnchangedSource()
