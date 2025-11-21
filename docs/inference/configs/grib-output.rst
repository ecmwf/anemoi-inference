.. _grib-output:

#############
 GRIB output
#############

.. warning::

   The GRIB output will be more efficient in conjunction with a
   :ref:`grib-input`. The GRIB output will use its input as template for
   encoding GRIB messages.

   If the input is not a GRIB file, the output will still attempt to
   encode the data as GRIB messages, but this is not always possible. In
   such cases, custom grib :ref:`templates` need to be provided.

.. note::

   This is placeholder documentation. This will explain how to use the
   GRIB output's behaviour can be configured.

The ``grib`` output can be used to specify many more options:

.. literalinclude:: yaml/grib-output_1.yaml

**********
 encoding
**********

A dictionary of key/value pairs to add to the encoding of the GRIB
messages.

****************
 check_encoding
****************

A boolean to check that the GRIB messages have been encoded correctly.

.. _templates:

***********
 templates
***********

anemoi-inference comes with a minimal set of built-in GRIB templates it
uses to write GRIB output. When you are running a model that runs on a
different grid and/or area than the built-in templates, or you need to
encode messages with a local definition, you may need to provide custom
templates.

Templates are configured by specifying template providers in the
``templates`` option of the grib output, for example:

.. code:: yaml

   output:
     grib:
       path: output.grib
       templates:
         - <provider>: <options>

The ``<options>`` can be omitted for providers that don't take any
options, be a string for providers that only take one option, or be a
dictionary for providers that take more than one option.

Multiple providers can be specified, and they will be tried in the order
they are listed. If a provider cannot provide a template for a variable,
the next provider will be tried. When all providers have been exhausted,
an error will be raised.

If the ``templates`` option is omitted and no template providers are
specified, the default providers are ``input`` and ``builtin`` (in that
order).

.. warning::

   The default providers are **not** automatically enabled once you
   specify any providers yourself. If you have custom providers, you
   need to explicitly include the ``input`` and/or ``builtin`` providers
   if you want to use them. For most use-cases with custom templates,
   it's recommended to also enable the ``input`` provider.

********************
 Template providers
********************

The following template providers are available:

``input``
=========

Use the messages from the GRIB input as templates. Only works with a
GRIB input.

.. code:: yaml

   output:
     grib:
       templates:
         - input

By default, only prognostic variables are taken from the input and
diagnostic variables will throw an error if no other template provider
can provide a template. A fallback mapping can optionally be provided to
map output variables to input variables:

.. code:: yaml

   output:
     grib:
       templates:
         - input:
             tp: 2t
             cp: 2t

In this example, if the output variables ``tp`` and ``cp`` are missing
from the input, the input template for ``2t`` will be used instead.

.. note::

   The fallback mapping only applies to output variables that are
   missing from the input. If the input contains an output variable, its
   template will always be used.

``builtin``
===========

Use the built-in templates that come with the package. By default these
will be used for diagnostic variables, or if the input is not GRIB. Only
a limited number of grids are `included
<https://github.com/ecmwf/anemoi-inference/blob/main/src/anemoi/inference/grib/templates/builtin.yaml>`_,
and only at a global area.

.. code:: yaml

   output:
     grib:
       templates:
         - builtin

``file``
========

Load templates from a specified GRIB file.

.. code:: yaml

   output:
     grib:
       templates:
         - file: /path/to/template.grib

By default, only the first message in the file will be used as template
for **all** output variables. This behaviour can be changed with the
following options:

-  ``path`` the path to the GRIB file

-  ``mode`` how to select a message from the grib file to use as
   template. Can be one of:

   -  ``first`` (default) use the first message in the file
   -  ``last`` use the last message in the file
   -  ``auto`` select variable from the file matching the output
      variable name

-  ``variables`` the output variable name(s) for which to use this
   template file (list or string). If empty, applies to all variables.

.. tip::

   A recommended use-case when using the GRIB input, is to use the
   ``input`` provider to cover prognostic variables, and use a ``file``
   provider in auto-mode for diagnostic variables:

   .. code:: yaml

      output:
        grib:
          templates:
            - input
            - file:
                path: /path/to/file-with-diagnostic-variables.grib
                mode: auto

``samples``
===========

Load templates from specified GRIB files based on rules matched against
a variable's metadata.

This provider takes a list of samples, each consisting of a dictionary
of matching rules and a path to a GRIB file. Whenever an output template
is requested, the sample's rules are checked against the output
variable's metadata. If all ``key:value`` pairs in the sample's rule
match the corresponding pair in the variable's metadata, the sample file
is selected. Samples are evaluated in the order they are listed.

.. warning::

   Only **the first message** in the sample GRIB file is used as
   template.

.. code:: yaml

   output:
     grib:
       templates:
         - samples:
           - - { matching rules 1 }
             - /path/to/template1.grib
           - - { matching rules 2 }
             - /path/to/template2.grib
           # etc

A practical use-case is to provide different templates for different
grids and/or levtypes. For example, if you are running models on both
the N320 and O96 grids, you could provide templates like this:

.. code:: yaml

   output:
     grib:
       templates:
         - samples:
           - - { grid: N320, levtype: pl }
             - /path/to/template-n320-pl.grib
           - - { grid: N320 }
             - /path/to/template-n320-sfc.grib
           - - { grid: O96, levtype: pl }
             - /path/to/template-o96-pl.grib
           - - { grid: O96 }
             - /path/to/template-o96-sfc.grib

Note that the sfc (surface) template doesn't have a levtype rule. This
way, the sfc template will be used for all variables that are not pl
(pressure level). It is also possible to have an empty rule ``{}`` as a
catch-all, but care needs to be taken as it may lead to incorrect
encoding if the grids do not match.

Common keys that can be used in the rules include: ``grid``,
``levtype``, ``param``

.. note::

   As soon as all keys in the sample rules match, the sample is
   selected. This makes the order of samples important: more specific
   rules should be listed first, and more general last.

.. tip::

   The sample file path can be a format string using metadata keys. The
   above example can be rewritten as:

   .. code:: yaml

      output:
        grib:
          templates:
            - samples:
              - - levtype: pl
                - /path/to/template-{grid}-pl.grib
              - - {}
                - /path/to/template-{grid}-sfc.grib

   Or even shorter:

   .. code:: yaml

      output:
        grib:
          templates:
            - samples:
              - - {}
                - /path/to/template-{grid}-{levtype}.grib

   But notice the subtle difference: in the first example, the sfc file
   is used for *all* non-pl variables. In the second example, a file
   must exist for each individual levtype.

.. tip::

   Samples can also be provided via a separate YAML file:

   .. code:: yaml

      output:
        grib:
          templates:
            - samples: /path/to/samples.yaml
