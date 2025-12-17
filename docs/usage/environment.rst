.. _usage-environment:

###################
 Environment Setup
###################

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

.. note::

   If using the `dataset` input source, you will need `anemoi-datasets
   <https://anemoi.readthedocs.io/projects/datasets/en/latest/>`_
   installed.

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

.. seealso::

   - :ref:`installing` - Installation instructions
   - :ref:`inspect-command` - Inspect checkpoint metadata and requirements
   - :ref:`usage-quickstart` - Quickstart guide
   - :ref:`usage-optimisation` - Performance optimisation tips
