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
