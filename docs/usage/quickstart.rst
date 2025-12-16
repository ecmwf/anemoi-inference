.. _usage-quickstart:

############
 Quickstart
############

This section provides a quickstart guide to using the
``anemoi-inference`` package. It covers the following topics:

.. contents:: Table of Contents
   :local:
   :depth: 1

anemoi-inference can be run via the command line interface (CLI) or
programmatically via Python scripts. This quickstart guide focuses on
the CLI usage. For more details on programmatic usage, please refer to
the :ref:`other api usage <api_introduction>` documentation.

***************
 Configuration
***************

The command line tool requires a configuration file in YAML format to
specify the inference settings. Below is an example of a minimal
configuration file:

.. literalinclude:: yaml/quickstart1.yaml
   :language: yaml

Primarily, the configuration file should specify:

-  The path to the trained model checkpoint.
-  The lead time for the forecast.
-  The input data source, which can be local files or remote data
   stores.
-  The output data destination.

For more details on the configuration options, please refer to the
:ref:`configuration reference <config_introduction>` documentation.

It is possible to run with a huggingface checkpoint directly by
specifying the checkpoint as follows:

.. literalinclude:: yaml/quickstart2.yaml
   :language: yaml

.. warning::

   This requires `huggingface_hub` to be installed in your environment.

   .. code:: bash

      pip install huggingface_hub

.. admonition:: Complete Example

   Therefore, a complete minimal configuration file for running
   inference with a huggingface checkpoint would look like this:

      .. literalinclude:: yaml/quickstart3.yaml

      .. warning::

         Additionally, to download data from the Copernicus Data Store,
         you need to have the `cdsapi` package installed and configured
         in your environment.
         https://cds.climate.copernicus.eu/how-to-api.

*************
 Environment
*************

It is recommended to create a new Python virtual environment for running
``anemoi-inference`` to isolate tasks within a ML workflow. However, it
is also recommended to ensure that the versions of key packages are
compatible / identical to those used during training.

This of particular importance for the following packages:

-  `anemoi-models
   <https://anemoi.readthedocs.io/projects/models/en/latest/>`_
-  `anemoi-graphs
   <https://anemoi.readthedocs.io/projects/graphs/en/latest/>`_
-  `torch <https://pytorch.org/>`_
-  `torch_geometric <https://pytorch-geometric.readthedocs.io/>`_

.. important::

   As ``anemoi`` is still in active development, it is recommended to at
   least use the same major and minor version of the above ``anemoi``
   packages as those used during training.

.. tip::

   You can check the versions of the packages used during training by
   inspecting the checkpoint metadata with the command and getting a
   list of requirements:

   .. code:: bash

      anemoi-inference inspect --requirements /path/to/inference-last.ckpt

**********************
 Running the forecast
**********************

Now, once you have your configuration file ready, and your environment
is set up, it is time to run the forecast!

You can run the inference using the command line tool as follows:

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

Some details of how to use the programmatic interface can be found on
the `hugging-face example for AIFS Single 1
<https://huggingface.co/ecmwf/aifs-single-1.1#how-to-get-started-with-the-model>`_.
