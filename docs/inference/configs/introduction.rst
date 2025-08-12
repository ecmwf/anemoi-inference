.. _config_introduction:

#########
 Configs
#########

This document provides an overview of the configuration to provide to
the :ref:`anemoi-inference run <run_command>` command line tool.

The configuration file is a YAML file that specifies various options. It
is extended by `OmegaConf <https://github.com/omry/omegaconf>`_ such
that `interpolations
<https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#variable-interpolation>`_
can be used. It is composed of :ref:`top level <top-level>` options
which are usually simple values such as strings, number or booleans. The
configuration also provide ways to specify which internal classes to use
for the :ref:`inputs <inputs>` and :ref:`outputs <outputs>`, and how to
configure them.

In that case, the general format is shown below. The first entry
(``mars`` or ``grib`` in the examples below) corresponds to the
underlying Python class that will be used to process the input, output
or any other polymorphic behaviour, followed by arguments specific to
that class.

.. literalinclude:: yaml/introduction_1.yaml
   :language: yaml

or:

.. literalinclude:: yaml/introduction_2.yaml
   :language: yaml

If the underlying class does not require any arguments, or you wish to
use the default parameters, then configuration can be simplified as:

.. literalinclude:: yaml/introduction_3.yaml
   :language: yaml

or if it expects a single argument, it can be simplified as:

.. literalinclude:: yaml/introduction_4.yaml
   :language: yaml

.. toctree::
   :maxdepth: 1
   :caption: Configurations

   top-level
   inputs
   outputs
   processors
   forcings
   icon-input
   grib-input
   grib-output
