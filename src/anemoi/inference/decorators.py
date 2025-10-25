# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path
from typing import Any
from typing import TypeVar

from anemoi.inference.context import Context

LOG = logging.getLogger("anemoi.inference")

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

    Also supports capturing all positional arguments into the main argument as a tuple:
    ```
    @main_argument("paths", capture_all_args=True)
    class MultiFileOutput
        def __init__(context, paths=None, archive_requests=None):
            ...
    output = MultiFileOutput(context, "out1.grib", "out2.grib")
    ```
    """

    def __init__(self, name: str, *, capture_all_args: bool = False):
        """Initialize the main_argument decorator.

        Parameters
        ----------
        name : str
            The name of the main argument.
        capture_all_args : bool, optional
            If True, captures all positional arguments into the main argument as a tuple.
        """
        self.name = name
        self._capture_all_args = capture_all_args

    def __call__(self, cls: F) -> F:
        """Decorate the class to set the main argument."""

        if not isinstance(cls, type):
            raise TypeError("'main_argument' can only be used to decorate classes")

        class WrappedClass(cls):
            def __init__(wrapped_cls, context: Context, main: object = MARKER, *args: Any, **kwargs: Any) -> Any:
                if main is not MARKER:
                    if self._capture_all_args:
                        kwargs[self.name] = (main,) + args
                        args = tuple()
                    else:
                        kwargs[self.name] = main
                super().__init__(context, *args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})


class ensure_path:
    """Decorator to ensure a path argument is a Path object and optionally exists.

    If `is_dir` is True, the path is treated as a directory, if not for files, the parent directory is treated as a directory.
    If `must_exist` is True, the directory must exist.
    If `create` is True, the directory will be created if it doesn't exist.

    For example:
    ```
    @ensure_path("dir", create=True)
    class GribOutput
        def __init__(context, dir=None, archive_requests=None):
            ...
    """

    def __init__(self, arg: str, is_dir: bool = False, create: bool = True, must_exist: bool = False):
        self.arg = arg
        self.is_dir = is_dir
        self.create = create
        self.must_exist = must_exist

    def __call__(self, cls: F) -> F:
        """Decorate the object to ensure the path argument is a Path object."""

        class WrappedClass(cls):
            def __init__(wrapped_cls, context: Context, *args: Any, **kwargs: Any) -> None:
                if self.arg not in kwargs:
                    LOG.debug(f"Argument '{self.arg}' not found in kwargs, cannot ensure path.")
                    super().__init__(context, *args, **kwargs)
                    return

                path = kwargs[self.arg] = Path(kwargs[self.arg])
                if not self.is_dir:
                    path = path.parent

                if self.must_exist:
                    if not path.exists():
                        raise FileNotFoundError(f"Path '{path}' does not exist.")
                if self.create:
                    path.mkdir(parents=True, exist_ok=True)

                super().__init__(context, *args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})


class ensure_dir(ensure_path):
    """Decorator to ensure a directory path argument is a Path object and optionally exists.

    If `must_exist` is True, the directory must exist.
    If `create` is True, the directory will be created if it doesn't exist.

    For example:
    ```
    @ensure_dir("dir", create=True)
    class PlotOutput
        def __init__(context, dir=None, ...):
            ...
    """

    def __init__(self, arg: str, create: bool = True, must_exist: bool = False):
        super().__init__(arg, is_dir=True, create=create, must_exist=must_exist)
