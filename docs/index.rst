.. _anemoi-inference:

.. _index-page:

##############################################
 Welcome to `anemoi-inference` documentation!
##############################################

.. warning::

   This documentation is work in progress.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

This package provides a series of utility functions for used by the rest
of the *Anemoi* packages.

-  :doc:`overview`
-  :doc:`cli/introduction`
-  :doc:`installing`

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   cli/introduction
   installing

*****************
 Recipe Examples
*****************

-  :doc:`getting-started/getting-started`

.. toctree::
   :maxdepth: 1
   :hidden:

   getting-started/getting-started



************
 User Guide
************

-  :doc:`inference/parallel`

   inference/parallel

*********************
 Tree levels of APIs
*********************

-  :doc:`apis/level1`
-  :doc:`apis/level2`
-  :doc:`apis/level3`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: APIs

   apis/level1
   apis/level2
   apis/level3

********************
 Command line tools
********************

-  :doc:`cli/introduction`
-  :doc:`cli/run`
-  :doc:`cli/metadata`
-  :doc:`cli/validate`
-  :doc:`cli/inspect`
-  :doc:`cli/patch`
-  :doc:`cli/requests`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Command line tool

   cli/introduction
   cli/run
   cli/metadata
   cli/validate
   cli/inspect
   cli/patch
   cli/requests

.. toctree::
   :maxdepth: 1
   :caption: Configurations

   configs/introduction
   configs/top-level
   configs/inputs
   configs/outputs
   configs/forcings
   configs/grib-input
   configs/grib-output

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Modules

   modules/runner
   modules/checkpoint
   modules/forcings
   modules/inputs
   modules/metadata
   modules/outputs

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developing Anemoi Graphs

   dev/contributing
   dev/code_structure
   dev/testing

*****************
 Anemoi packages
*****************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html
