# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from io import IOBase
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from anemoi.inference.context import Context

from .gribfile import GribIoOutput

LOG = logging.getLogger(__name__)


class GribMemoryOutput(GribIoOutput):
    """Handles grib files in memory."""

    def __init__(
        self,
        context: Context,
        *,
        out: IOBase,
        encoding: Optional[Dict[str, Any]] = None,
        archive_requests: Optional[Dict[str, Any]] = None,
        check_encoding: bool = True,
        templates: Optional[Union[List[str], str]] = None,
        grib1_keys: Optional[Dict[str, Any]] = None,
        grib2_keys: Optional[Dict[str, Any]] = None,
        modifiers: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        """Initialize the GribFileOutput.

        Parameters
        ----------
        context : Context
            The context.
        out : IOBase
            Output stream or file-like object for writing GRIB data.
        encoding : dict, optional
            The encoding dictionary, by default None.
        archive_requests : dict, optional
            The archive requests dictionary, by default None.
        check_encoding : bool, optional
            Whether to check encoding, by default True.
        templates : list or str, optional
            The templates list or string, by default None.
        grib1_keys : dict, optional
            The grib1 keys dictionary, by default None.
        grib2_keys : dict, optional
            The grib2 keys dictionary, by default None.
        modifiers : list, optional
            The list of modifiers, by default None.
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        variables : list, optional
            The list of variables, by default None.
        """
        super().__init__(
            context,
            out=out,
            encoding=encoding,
            archive_requests=archive_requests,
            check_encoding=check_encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
            variables=variables,
            split_output=False,
        )
