# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from functools import wraps

MARKER = object()


class main_argument:
    """Decorator to set the main argument of a function

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

    def __init__(self, name):
        self.name = name

    def __call__(self, f):

        @wraps(f)
        def decorator(context, main=MARKER, *args, **kwargs):
            if main is not MARKER:
                kwargs[self.name] = main
            return f(context, *args, **kwargs)

        return decorator
