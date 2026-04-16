# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import TypeVar

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata

LOG = logging.getLogger("anemoi.inference")

F = TypeVar("F", bound=type)
UNIQUE_PATHS = defaultdict(set)


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

        # position of the main argument changes depending on whether the decorated class takes `metadata` or not
        # so inspect the wrapped class to find the offset of the main arguments in the args list
        for klass in cls.mro():
            # our decorators can be stacked, so traverse the MRO to find the parent decorated class
            parameters = inspect.signature(klass.__init__).parameters
            if "wrapped_cls" in parameters:
                continue
            _offset = 2 if "metadata" in parameters else 1  # accounts for `self``
            break

        class WrappedClass(cls):
            def __init__(wrapped_cls, *args, **kwargs) -> None:
                args = list(args)
                if len(args) > _offset:
                    kwargs[self.name] = args.pop(_offset)
                super().__init__(*args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})


class ensure_path:
    """Decorator to ensure a path argument is a Path object and optionally exists.

    If `is_dir` is True, the path is treated as a directory, if not for files, the parent directory is treated as a directory.
    If `must_exist` is True, the directory must exist.
    If `create` is True, the directory will be created if it doesn't exist.
    If 'unique' is True, the same path cannot be reused between multiple decorated classes.

    For example:
    ```
    @ensure_path("dir", create=True)
    class GribOutput
        def __init__(context, dir=None, archive_requests=None):
            ...
    """

    def __init__(
        self, arg: str, is_dir: bool = False, create: bool = True, must_exist: bool = False, unique: bool = True
    ):
        self.arg = arg
        self.is_dir = is_dir
        self.create = create
        self.must_exist = must_exist
        self.unique = unique

    def __call__(self, cls: F) -> F:
        """Decorate the object to ensure the path argument is a Path object."""

        class WrappedClass(cls):
            def __init__(wrapped_cls, *args: Any, **kwargs: Any) -> None:
                if self.arg not in kwargs:
                    LOG.debug(f"Argument '{self.arg}' not found in kwargs, cannot ensure path.")
                    super().__init__(*args, **kwargs)
                    return

                path = kwargs[self.arg] = Path(kwargs[self.arg])

                if self.unique:
                    if path in UNIQUE_PATHS[self.arg]:
                        raise ValueError(
                            f"'{self.arg}={path}' is already used by another output. For multi-dataset output, ensure you are using different output paths for each dataset."
                        )
                    UNIQUE_PATHS[self.arg].add(path)

                if not self.is_dir:
                    path = path.parent

                if self.must_exist:
                    if not path.exists():
                        raise FileNotFoundError(f"Path '{path}' does not exist.")
                if self.create:
                    path.mkdir(parents=True, exist_ok=True)

                super().__init__(*args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})


class ensure_dir(ensure_path):
    """Decorator to ensure a directory path argument is a Path object and optionally exists.

    If `must_exist` is True, the directory must exist.
    If `create` is True, the directory will be created if it doesn't exist.
    If 'unique' is True, the same path cannot be reused between multiple decorated classes.

    For example:
    ```
    @ensure_dir("dir", create=True)
    class PlotOutput
        def __init__(context, dir=None, ...):
            ...
    """

    def __init__(self, arg: str, create: bool = True, must_exist: bool = False, unique: bool = True):
        super().__init__(arg, is_dir=True, create=create, must_exist=must_exist, unique=unique)


class format_dataset_name:
    """Decorator to format a string argument with the dataset name.
    Substitutes `{dataset}` or `{dataset_name}` in the argument with the dataset name.
    Can only be used for classes that take `metadata`. For example:
    ```
    output:
        grib: output-{dataset}.grib
    ```
    """

    def __init__(self, arg: str):
        self.arg = arg

    def __call__(self, cls: F) -> F:
        if not isinstance(cls, type):
            raise TypeError(f"`{self.__class__.__name__}` can only be used to decorate classes")

        if not any("metadata" in inspect.signature(klass.__init__).parameters for klass in cls.mro()):
            raise TypeError(f"`{self.__class__.__name__}` can only be used to decorate classes that take `metadata`")

        class DefaultFormat(dict):
            def __missing__(self, key):
                return f"{{{key}}}"  # if the key is not found, return the placeholder unchanged

        class WrappedClass(cls):
            def __init__(wrapped_cls, context: Context, metadata: Metadata, *args: Any, **kwargs: Any) -> Any:
                assert self.arg in kwargs, f"{self.arg} not found in decorated class arguments: {kwargs}"
                assert isinstance(
                    kwargs[self.arg], str
                ), f"{self.arg} must be a string to use `{self.__class__.__name__}` decorator"

                name = metadata.dataset_name
                kwargs[self.arg] = kwargs[self.arg].format_map(DefaultFormat(dataset=name, dataset_name=name))
                super().__init__(context, metadata, *args, **kwargs)

        return type(cls.__name__, (WrappedClass,), {})
