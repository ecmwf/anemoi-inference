# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import re

_TEMPLATE_EXPRESSION_PATTERN = re.compile(r"\{(.*?)\}")


def render_template(template: str, handle: dict) -> str:
    """Render a template string with the given keyword arguments.

    Given a template string such as '{dateTime}_{step:03}.grib' and
    the GRIB handle, this function will replace the expressions in the
    template with the corresponding values from the handle, formatted
    according to the optional format specifier.

    For example, the template '{dateTime}_{step:03}.grib' with a handle
    containing 'dateTime' as '202501011200' and 'step' as 6 will
    produce '202501011200_006.grib'.

    Parameters
    ----------
    template : str
        The template string to render.
    handle : dict
        The dictionary to use for rendering the template.

    Returns
    -------
    str
        The rendered template string.
    """
    expressions = _TEMPLATE_EXPRESSION_PATTERN.findall(str(template))
    expr_format = [el.split(":") if ":" in el else [el, ""] for el in expressions]
    keys = {k[0]: format(handle.get(k[0]), k[1]) for k in expr_format}
    path = str(template).format(**keys)
    return path
