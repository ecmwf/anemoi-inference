.. _style-guide:

##############################
 Anemoi-Inference Style Guide
##############################

.. note::

   This style guide augements any existing general Anemoi style guides,
   such as the :ref:`contributing guide <dev-contributing>` or the
   :ref:`general Anemoi style guide <anemoi-docs:code-style>`. If this
   guide conflicts with wider project guidance in any way, follow the
   general guidance (and let us know so we can update these docs!)

********
 Typing
********

All code should include `type-hints
<https://docs.python.org/3/library/typing.html>`_. These type hints
should be reasonably restrictive, rather than using ``Any``.

If you need more complex type hints, these can be created in
``src/anemoi/inference/types.py`` and shared between different classes.
For example, if you can accept any of a string, list, or dictionary,
specify this by creating a union type in that class (or using an
existing one, if possible).

***************
 Documentation
***************

Docstrings use the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ format, in
line with the :ref:`general Anemoi documentation guidelines
<anemoi-docs:documentation-guidelines>`. Document the parameters, the
return value and any exceptions raised.

``docsig`` runs as a pre-commit hook and checks docstrings against the
function signature, so keep parameter names and their order in sync with
the code. Anything a user can set from the configuration file should
also be documented under ``docs/inference/configs/``.

***************
 General style
***************

This code base follows a :ref:`registry pattern
<dev-codebase-overview>`. (For more
details about code layout, read
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
