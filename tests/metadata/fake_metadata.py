class FakeMetadata:
    def __getattr__(self, name):
        return None
