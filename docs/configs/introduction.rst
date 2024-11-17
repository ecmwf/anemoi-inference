##############
 Introduction
##############

This document provides an overview of the configuration to provide to
the :ref:`anemoi-inference run <run-cli>` command line tool.

The configuration file is a YAML file that specifies various options. It
is composed of :ref:`top level <top-level>` options which are usially
simple values such as strings, number or booleans. The configuration
also provide ways to specify which internal classes to use for the
:ref:`inputs <inputs>` and :ref:`outputs <outputs>`, and how to
configure them.

In that case, the configuration as the format as show below. The key
``kind`` corresponds to and underlying Python class that will be used to
process the input, output or any other polymorphic behaviour.

.. literalinclude:: introduction_1.yaml
   :language: yaml

If the underlying class does not require any arguments, or you wish to
use the default parameters, you can use the following format:

.. literalinclude:: introduction_2.yaml
   :language: yaml

For example, the following configuration specifies the use of the
``grib`` input class, from which to read the initial conditions, and the
``printer`` output class, which prints the minimum and maximum values of
a few fields at each forecasting time step.

.. literalinclude:: introduction_3.yaml
   :language: yaml
