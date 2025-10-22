import pytest

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

    assert isinstance(main_argument("path")(_Cls), type)

    with pytest.raises(TypeError):
        main_argument("path")(lambda x: x)


def test_main_argument_with_capture_all():
    @main_argument("paths", capture_all_args=True)
    class _Cls:
        def __init__(self, context, paths=None):
            self.context = context
            self.paths = paths

    cls = _Cls("context", "path1", "path2")

    assert cls.context == "context"
    assert cls.paths == ("path1", "path2")

    assert isinstance(main_argument("paths", capture_all_args=True)(_Cls), type)
