.. _dev-contributing:

####################
 General guidelines
####################

Thank you for your interest in Anemoi Inference! Please follow the
:ref:`general Anemoi contributing guidelines
<anemoi-docs:contributing>`.

These include general guidelines for contributions to Anemoi,
instructions on setting up a development environment, and guidelines on
collaboration on GitHub, writing documentation, testing, and code style.

************
 Unit tests
************

Anemoi-inference includes unit tests that can be executed locally using
pytest. For more information on testing, please refer to the
:ref:`general Anemoi testing guidelines
<anemoi-docs:testing-guidelines>`.

****************************
 Classes and code structure
****************************

This code base follows a :ref:`registry pattern
<dev-codebase-overview>`. (For more details about code layout, read
:ref:`dev-add-new-configuration-option`). New features should be
contained in small classes, as a subclass to a larger, more general
case.

Always be aware when accepting user input (in our case, from the
configuration file) and ensure it is properly validated. If you are
enforcing specific expectations on structure, those should be clearly
documented in docstrings and in user documentation, with clear error
messages if the user does not meet them. Consider validating it using a
Pydantic model, which provides specific tooling for validation. These
are used in some places in the code base.

All code should include `type-hints
<https://docs.python.org/3/library/typing.html>`_. These type hints
should be reasonably restrictive, rather than using ``Any``.

If you need more complex type hints, these can be created in
``src/anemoi/inference/types.py`` and shared between different classes.
For example, if you can accept any of a string, list, or dictionary,
specify this by creating a union type in that class (or using an
existing one, if possible).
