.. _usage-quickstart:

############
 Quickstart
############

This page provides a quickstart guide to using the ``anemoi-inference``
package. It covers the following topics:

.. contents:: Table of Contents
   :local:
   :depth: 1

If you would like more information on what ``anemoi-inference`` does,
and how it fits within the ``anemoi`` ecosystem, see the
:ref:`introduction <index-page>` page.

``anemoi-inference`` can be run via the command line interface (CLI) or
programmatically via Python scripts. This quickstart guide focuses on
the CLI usage.

Throughout this guide we assume you have run :ref:`anemoi-training
<anemoi-training:index-page>` and have a checkpoint on disk you are
ready to use. Additionally, we assume you have access to the training
dataset.

***************
 Configuration
***************

The CLI requires a configuration file in YAML format to specify the
inference options. We shall use the test dataset for initial conditions
here, to keep things simple. Below is an example of a minimal
configuration file:

.. literalinclude:: yaml/quickstart1.yaml
   :language: yaml

Primarily, the configuration file should specify:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   -  -  key
      -  Description

   -  -  `ckpt`
      -  Path to the trained model checkpoint.

   -  -  `lead_time`
      -  Lead time (in hours) for the forecast.

   -  -  `date`
      -  Initial condition date

   -  -  `input`
      -  Data source for initial conditions, which can be local files or
         remote data stores.

   -  -  `output`
      -  Output data destination.

For more details on the configuration options, please refer to the
:ref:`configuration reference <config_introduction>` documentation.

*************
 Environment
*************

For this guide, feel free to reuse your training environment as it will
keep things simple. For other use cases, the recommendation is to create
a new virtual environment. If you decide to do so, please check out the
:ref:`environment setup <usage-environment>` page, as it is important
that you install the correct versions of the ``anemoi`` packages.

**********************
 Running the forecast
**********************

Now, once you have your configuration file ready, and your environment
is set up, it is time to run the forecast!

You can run the inference using the :ref:`run <run-command>` command
line tool as follows, which more information can be found in the
:ref:`advanced cli <usage-advanced-cli>` documentation.

.. code:: bash

   anemoi-inference run /path/to/inference_config.yaml

Viola! Now as the model runs forward in inference, you should see output
information being printed to the terminal.

************
 Next steps
************

Now that you have run your first inference, you may want to explore more
the next steps of using ``anemoi-inference``. The following sections
provide more details on various next topics:

.. toctree::
   :maxdepth: 1

   advanced/cli
   advanced/sources
   advanced/saving
   advanced/remote-checkpoints

****************
 Advanced usage
****************

For more advanced usage, including running in parallel on a cluster,
using external graphs, and detailed configuration options, please refer
to the following sections:

.. toctree::
   :maxdepth: 1

   ../inference/parallel
   ../inference/external-graph
   ../inference/configs/introduction

Some details on how to use the programmatic interface of
``anemoi-inference`` can be found on the `hugging-face example for AIFS
Single 1
<https://huggingface.co/ecmwf/aifs-single-1.1#how-to-get-started-with-the-model>`_.
