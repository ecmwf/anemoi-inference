import pytest

from anemoi.inference.decorators import format_dataset_name
from anemoi.inference.decorators import main_argument


def test_main_argument():
    @main_argument("path")
    class _Cls:
        def __init__(self, context, metadata, flag=True, path=None):
            self.context = context
            self.metadata = metadata
            self.flag = flag
            self.path = path

    cls = _Cls("context", "metadata", "path")

    assert cls.context == "context"
    assert cls.metadata == "metadata"
    assert cls.flag is True
    assert cls.path == "path"

    assert isinstance(main_argument("path")(_Cls), type)  # decorator should return a class

    with pytest.raises(TypeError):
        main_argument("path")(lambda x: x)  # not a class

    # no metadata
    @main_argument("path")
    class _NoMetadataCls:
        def __init__(self, context, flag=True, path=None):
            self.context = context
            self.flag = flag
            self.path = path

    cls = _NoMetadataCls("context", "path")

    assert cls.context == "context"
    assert cls.flag is True
    assert cls.path == "path"

    # stacking of decorators
    @main_argument("path")
    @main_argument("path")
    class _StackedCls(_Cls):
        pass

    cls = _StackedCls("context", "metadata", "path")

    assert cls.context == "context"
    assert cls.flag is True
    assert cls.path == "path"


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

    cls = _Cls("context", metadata, path="output-{dataset}.grib")
    assert cls.path == "output-era5.grib"

    cls = _Cls("context", metadata, path="output-{dataset}-{levtype}.grib")
    assert cls.path == "output-era5-{levtype}.grib"

    cls = _Cls("context", metadata, path="output-{dataset_name}.grib")
    assert cls.path == "output-era5.grib"

    cls = _Cls("context", metadata, path="output.grib")
    assert cls.path == "output.grib"

    assert isinstance(format_dataset_name("path")(_Cls), type)  # decorator should return a class

    with pytest.raises(TypeError) as excinfo:
        format_dataset_name("path")(lambda x: x)  # not a class
    assert "only be used to decorate classes" in str(excinfo.value)

    with pytest.raises(AssertionError):
        _Cls("context", metadata)  # missing path

    with pytest.raises(AssertionError):
        _Cls("context", metadata, path=42)  # path is not a string

    # no metadata argument
    with pytest.raises(TypeError) as excinfo:

        @format_dataset_name("path")
        class _NoMetadataCls:
            def __init__(self, context, *, path=None):
                pass

    assert "only be used to decorate classes that take `metadata`" in str(excinfo.value)
