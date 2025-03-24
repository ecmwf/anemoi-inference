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

**************
 About Anemoi
**************

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

****************
 Quick overview
****************

The anemoi-inference package provides a framework for running inference
with data-driven weather forecasting models within the Anemoi ecosystem.
It is designed to efficiently handle model execution and streamline
input data processing.

anemoi-inference offers a high-level interface that integrates
seamlessly with trained machine learning models. The package allows you
to:

-  Load and preprocess input data from anemoi-datasets, ensuring
   compatibility with the trained model.
-  Run inference using machine learning-based weather forecasting
   models.
-  Save and manage forecast outputs in a variety of formats.
-  Run inference tasks either using programmatic or via a command-line
   APIs.

Inference configurations are specified using a YAML file, which defines
model parameters, input datasets, and output formats. The command-line
tool allows users to run inference tasks, inspect results, and manage
forecast outputs. In the rest of this documentation, you will learn how
to configure and execute inference workflows using anemoi-inference. A
complete example of running a forecast with a trained model can be found
in the :ref:` Generating your first forecast <usage-getting-started>`
section.

************
 Installing
************

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-inference

Get more information in the :ref:`installing <installing>` section.

**************
 Contributing
**************

.. code:: bash

   git clone ...
   cd anemoi-inference
   pip install .[dev]

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc

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

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   cli/introduction
   installing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Recipe Examples

   usage/getting-started

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using Anemoi Inference

   inference/parallel
   inference/apis/introduction
   inference/configs/introduction

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
   :hidden:
   :caption: Modules

   modules/runner
   modules/checkpoint
   modules/forcings
   modules/inputs
   modules/metadata
   modules/outputs
   modules/processor

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developing Anemoi Inference

   dev/contributing
   dev/code_structure
   dev/testing
