# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..context import Context
from ..decorators import main_argument
from ..metadata import Metadata
from ..types import Date
from ..types import State
from . import input_registry
from .ekd import EkdInput

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import earthkit.data as ekd


@input_registry.register("opendap")
@main_argument("url")
class OpenDAPInput(EkdInput):
    """Handles OpenDAP sourced files."""

    trace_name = "OpenDAP file"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        url: str | list[str],
        **kwargs,
    ) -> None:
        """Initialise the OpenDAPInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        url: str | list[str]
            The URL or list of URL's of the OpenDAP file / server.
            Python format strings can be used to dynamically construct the URL, with the following variables available:
            - {date} : The date for which the input is being created.
        """
        super().__init__(context, metadata, **kwargs)
        self.url = url if isinstance(url, list) else [url]

    def _resolve_url(self, date: Date | None) -> list[str]:
        """Resolve the URL for the given date."""
        if date is None:
            if any("{date" in u for u in self.url):
                raise ValueError("Date must be provided to resolve the URL.")
            return self.url
        return [u.format(date=date) for u in self.url]

    def _retrieve_from_opendap(self, resolved_url: list[str]) -> "ekd.FieldList":
        """Retrieve the data from the OpenDAP server, filtering to the first valid_datetime if multiple present."""
        import earthkit.data as ekd

        retrieved_data = [ekd.from_source("opendap", url) for url in resolved_url]
        combined_fieldlist = ekd.FieldList.from_fields([f for fl in retrieved_data for f in fl])  # type: ignore[reportGeneralTypeIssues]
        if len(combined_fieldlist.unique_values("valid_datetime")["valid_datetime"]) > 1:
            LOG.warning(
                f"Retrieved data from OpenDAP server has multiple valid_datetimes: {combined_fieldlist.unique_values('valid_datetime')}. Using the first one."
            )
            combined_fieldlist = combined_fieldlist.isel(valid_datetime=0)
        return combined_fieldlist  # type: ignore[reportReturnType]

    def create_input_state(self, *, date: Date | None, ref_date_index: int = -1, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        ref_date_index : int = -1
            If 0 takes the first date, if -1 takes the last date in sequence.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        State
            The created input state.
        """
        import earthkit.data as ekd

        date = np.datetime64(date).astype(datetime)
        dates = [date + h for h in self.metadata.lagged]

        fieldlists = []

        for d in dates:
            resolved_url = self._resolve_url(d)
            LOG.info(f"Retrieving data for input_state from OpenDAP server: {resolved_url}")
            fieldlists.append(self._retrieve_from_opendap(resolved_url))

        fieldlist = ekd.FieldList.from_fields([f for fl in fieldlists for f in fl])
        return self._create_input_state(fieldlist, date=date, ref_date_index=ref_date_index, **kwargs)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        dates : List[Date]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        State
            The loaded forcings state.
        """
        fieldlists = []

        for d in dates:
            resolved_url = self._resolve_url(d)
            LOG.info(f"Retrieving data for forcings from OpenDAP server: {resolved_url}")
            fieldlists.append(self._retrieve_from_opendap(resolved_url))

        fieldlist = ekd.FieldList.from_fields([f for fl in fieldlists for f in fl])
        return self._load_forcings_state(fieldlist, dates=dates, current_state=current_state)
