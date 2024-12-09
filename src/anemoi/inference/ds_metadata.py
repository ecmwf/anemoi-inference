
from anemoi.inference.metadata import Metadata

import logging
import os
import warnings
from collections import defaultdict
from functools import cached_property

import numpy as np
from anemoi.transform.variables import Variable
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from icecream import ic

class Metadata_0(Metadata):
    """Metadata class for the first input dataset"""
    def __init__(self, metadata, supporting_arrays={}):
        super().__init__(metadata, supporting_arrays)
        self._indices.data.input = self._indices.data.input_0
        self._indices.model.input = self._indices.model.input_0

    @cached_property
    def number_of_input_grid_points(self):
        """Return the number of grid points per fields"""
        try:
            return self._metadata.dataset.shape[0][0][-1]
        except AttributeError:
            return self._legacy_number_of_grid_points()

    
class Metadata_1(Metadata):
    """Metadata class for the second input dataset"""   
    def __init__(self, metadata, supporting_arrays={}):
        super().__init__(metadata, supporting_arrays)
        self._indices.data.input = self._indices.data.input_1
        self._indices.model.input = self._indices.model.input_1

    @cached_property
    def number_of_input_grid_points(self):
        """Return the number of grid points per fields"""
        try:
            return self._metadata.dataset.shape[0][1][-1]
        except AttributeError:
            return self._legacy_number_of_grid_points()

    @property
    def variables(self):
        return tuple(
            self._metadata.dataset["arguments"]["args"][0]["dataset"]["x"]["zip"][1][
                "select"
            ]
        )

    def variable_categories(self):
        result = defaultdict(set)
        typed_variables = self.typed_variables

        variables_in_data_space = self.variables
        ic(variables_in_data_space)
        
   
        for name in self._config.data.forcing:
            result[name].add("forcing")

        for idx in self._indices.data.input.diagnostic:
            name = variables_in_data_space[idx]
            result[name].add("diagnostic")

        # assert self._indices.model.input.prognostic == self._indices.model.output.prognostic

        """
        variables_in_model_space = self.output _tensor_index_to_variable
        ic(variables_in_model_space)
        for idx in self._indices.model.input.prognostic:
            name = variables_in_model_space[idx]
            result[name].add("prognostic")
        """
            
        for name, v in typed_variables.items():
            if v.is_accumulation:
                result[name].add("accumulation")

            if v.is_constant_in_time:
                result[name].add("constant")

            if v.is_computed_forcing:
                result[name].add("computed")

        for name in self.variables:
            if name not in result:
                raise ValueError(f"Variable {name} has no category")

            result[name] = sorted(result[name])
        
        ic(result)

        return result
