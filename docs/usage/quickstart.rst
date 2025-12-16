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
the CLI usage. For more details on programmatic usage, please refer to
the :ref:`other api usage <api_introduction>` documentation.

***************
 Configuration
***************

The CLI requires a configuration file in YAML format to specify the
inference options. Below is an example of a minimal configuration file:

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

.. admonition:: Remote Checkpoints

   It is possible to run with a checkpoint stored in huggingface
   directly by specifying the checkpoint as follows:

   .. literalinclude:: yaml/quickstart2.yaml
      :language: yaml

   .. warning::

      To use huggingface stored models requires `huggingface_hub
      <https://github.com/huggingface/huggingface_hub>`_ to be installed
      in your environment.

      .. code:: bash

         pip install huggingface_hub

Complete example
================

Therefore, a complete minimal configuration file for running inference
with a huggingface checkpoint would look like this:

.. literalinclude:: yaml/quickstart3.yaml
   :language: yaml

.. warning::

   To download data from the Copernicus Data Store (cds), you need to
   have the ``cdsapi`` package installed and configured in your
   environment. See `here
   <https://cds.climate.copernicus.eu/how-to-api>`_ for more
   information.

.. admonition:: OpenData

   As CDS requires an account, some users may find it easier to use the
   `opendata` service from ECMWF to initialise the model. This data is
   openly available for the last three days under a permissive license.

   To install the plugin which provides access to the `opendata`
   service:

   .. code:: bash

      pip install anemoi-plugins-ecmwf-inference[opendata]

   To use this input simply reference it by name in the input block:

   .. code:: yaml

      input: opendata

   .. warning::

      Initial conditions are only available for the past three days
      using the `opendata` service.

*************
 Environment
*************

..
   Duplicated into the environment page for emphasis ---

It is recommended to :ref:`create <installing>` a new Python virtual
environment for running ``anemoi-inference`` to isolate tasks within a
ML workflow. When creating this environment it is also recommended to
ensure that the versions of key packages are compatible / identical to
those used during training.

This is of particular importance for the following packages:

-  `anemoi-models
   <https://anemoi.readthedocs.io/projects/models/en/latest/>`_
-  `anemoi-graphs
   <https://anemoi.readthedocs.io/projects/graphs/en/latest/>`_
-  `torch <https://pytorch.org/>`_
-  `torch_geometric <https://pytorch-geometric.readthedocs.io/>`_

.. important::

   As ``anemoi`` is still in active development, it is recommended to
   use the same version of the above ``anemoi`` packages as those used
   during training.

.. tip::

   You can check the versions of the packages used during training by
   inspecting the checkpoint metadata with the :ref:`inspect
   <inspect-command>` command and getting a list of requirements:

   .. code:: bash

      anemoi-inference inspect --requirements /path/to/inference-last.ckpt

**********************
 Running the forecast
**********************

Now, once you have your configuration file ready, and your environment
is set up, it is time to run the forecast!

You can run the inference using the :ref:`run <run-command>` command
line tool as follows, which more information can be found at
:ref:`api_level3`

.. code:: bash

   anemoi-inference run /path/to/inference_config.yaml

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

Additionally, you can explore the various APIs provided by
``anemoi-inference`` for programmatic usage:

.. toctree::
   :maxdepth: 1

   ../usage/apis/level1
   ../usage/apis/level2
   ../usage/apis/level3

Some details of how to use the programmatic interface can be found on
the `hugging-face example for AIFS Single 1
<https://huggingface.co/ecmwf/aifs-single-1.1#how-to-get-started-with-the-model>`_.
