# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from functools import wraps
from typing import Any
from typing import Callable

from anemoi.inference.context import Context

MARKER = object()


class main_argument:
    """Decorator to set the main argument of a function.

    For example...

    @main_argument("path")
    def grib_file_output(context, path, encoding=None, archive_requests=None):
        ...

    So we can have:

    output:
        grib: out.grib

    means the same as

    output:
        grib:
            path: out.grib
    """

    def __init__(self, name: str):
        """Initialize the main_argument decorator.

        Parameters
        ----------
        name : str
            The name of the main argument.
        """
        self.name = name

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorate the function to set the main argument.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to decorate.

        Returns
        -------
        Callable[..., Any]
            The decorated function.
        """

        @wraps(func)
        def wrapped(context: Context, main: object = MARKER, *args: Any, **kwargs: Any) -> Any:
            """Decorator function to set the main argument.

            Parameters
            ----------
            context : Any
                The context in which the function is called.
            main : Any, optional
                The main argument value, by default MARKER.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            Any
                The result of the decorated function.
            """
            if main is not MARKER:
                kwargs[self.name] = main

            return func(context, *args, **kwargs)

        return wrapped
