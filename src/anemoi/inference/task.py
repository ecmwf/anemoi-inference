# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from abc import ABC

LOG = logging.getLogger(__name__)


class Task(ABC):
    """Abstract base class for tasks.

    Parameters
    ----------
    name : str
        The name of the task.
    """

    def __init__(self, name: str) -> None:
        """Initialize the Task.

        Parameters
        ----------
        name : str
            The name of the task.
        """
        self.name = name

    def __repr__(self) -> str:
        """Return a string representation of the Task.

        Returns
        -------
        str
            String representation of the Task.
        """
        return f"{self.__class__.__name__}({self.name})"
