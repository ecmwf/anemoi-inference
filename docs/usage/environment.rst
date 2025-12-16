.. _usage-environment:

..
   Duplicated from the quickstart for emphasis ---

#############
 Environment
#############

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
