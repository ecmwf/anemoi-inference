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

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig

from .gribfile import GribIoOutput

LOG = logging.getLogger(__name__)


class GribMemoryOutput(GribIoOutput):
    """Handles grib files in memory."""

    def __init__(
        self,
        context: Context,
        *,
        out: IOBase,
        post_processors: list[ProcessorConfig] | None = None,
        encoding: dict[str, Any] | None = None,
        archive_requests: dict[str, Any] | None = None,
        check_encoding: bool = True,
        templates: list[str] | str | None = None,
        grib1_keys: dict[str, Any] | None = None,
        grib2_keys: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Initialize the GribFileOutput.

        Parameters
        ----------
        context : Context
            The context.
        out : IOBase
            Output stream or file-like object for writing GRIB data.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
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
            post_processors=post_processors,
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
