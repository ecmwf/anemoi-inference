# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import TypeVar

from anemoi.inference.context import Context

MARKER = object()
F = TypeVar("F", bound=type)


class main_argument:
    """Decorator to set the main argument of a class. Only for classes with a 'context' argument.

    For example:
    ```
    @main_argument("path")
    class GribOutput
        def __init__(context, encoding=None, path=None, archive_requests=None):
            ...
    output = GribOutput(context, "out.grib")
    ```
    So in the config we can have:
    ```
    output:
        grib: out.grib
    ```
    meaning the same as
    ```
    output:
        grib:
            path: out.grib
    ```
    """

    def __init__(self, name: str):
        """Initialize the main_argument decorator.

        Parameters
        ----------
        name : str
            The name of the main argument.
        """
        self.name = name

    def __call__(self, cls: F) -> F:
        """Decorate the class to set the main argument."""

        if not isinstance(cls, type):
            raise TypeError("'main_argument' can only be used to decorate classes")

        class WrappedClass(cls):
            def __init__(wrapped_cls, context: Context, main: object = MARKER, *args: Any, **kwargs: Any) -> Any:
                if main is not MARKER:
                    kwargs[self.name] = main
                super().__init__(context, *args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})
