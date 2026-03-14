import pytest

from anemoi.inference.decorators import format_dataset_name
from anemoi.inference.decorators import main_argument


def test_main_argument():
    @main_argument("path")
    class _Cls:
        def __init__(self, context, flag=True, path=None):
            self.context = context
            self.flag = flag
            self.path = path

    cls = _Cls("context", "path")

    assert cls.context == "context"
    assert cls.flag is True
    assert cls.path == "path"

    assert isinstance(main_argument("path")(_Cls), type)  # decorator should return a class

    with pytest.raises(TypeError):
        main_argument("path")(lambda x: x)  # not a class


def test_format_dataset_name():
    @format_dataset_name("path")
    class _Cls:
        def __init__(self, context, metadata, *, path=None):
            self.context = context
            self.path = path
            self.metadata = metadata

    class _Metadata:
        dataset_name = "era5"

    metadata = _Metadata()

    cls = _Cls("context", path="output-{dataset}.grib", metadata=metadata)
    assert cls.path == "output-era5.grib"

    cls = _Cls("context", path="output-{dataset_name}.grib", metadata=metadata)
    assert cls.path == "output-era5.grib"

    cls = _Cls("context", path="output.grib", metadata=metadata)
    assert cls.path == "output.grib"

    assert isinstance(format_dataset_name("path")(_Cls), type)  # decorator should return a class

    with pytest.raises(TypeError):
        format_dataset_name("path")(lambda x: x)  # not a class

    with pytest.raises(AssertionError):
        _Cls("context", path="output-{dataset}.grib")  # missing metadata

    with pytest.raises(AssertionError):
        _Cls("context", metadata=metadata)  # missing path

    with pytest.raises(AssertionError):
        _Cls("context", path=42, metadata=metadata)  # path is not a string
