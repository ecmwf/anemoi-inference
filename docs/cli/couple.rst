.. _couple-command:

###############
 Couple Command
###############

The ``couple`` command allows you to run multiple coupled inference tasks
that exchange data during execution. This is useful for running coupled
model systems (e.g., atmosphere-ocean coupling) or ensemble systems where
different model components need to communicate.

************
 Description
************

The ``couple`` command coordinates multiple inference tasks that run
simultaneously and exchange data through a specified transport mechanism.
Each task can be configured independently while sharing common settings
like date and lead time.

The command reads a coupling configuration file that defines:

- Individual tasks and their configurations
- Transport mechanism for data exchange between tasks
- Coupling relationships specifying which tasks exchange what data
- Global settings applied to all tasks

.. argparse::
   :module: anemoi.inference.__main__
   :func: create_parser
   :prog: anemoi-inference
   :path: couple

***************
 Configuration
***************

The coupling configuration file should have the following structure:

.. literalinclude:: yaml/couple1.yaml
   :language: yaml

Transport Options
=================

The transport mechanism controls how tasks communicate:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Transport
     - Description

   * - ``threads``
     - Run tasks in separate threads (lowest overhead, shared memory)

   * - ``processes``
     - Run tasks in separate processes (isolated memory)

   * - ``mpi``
     - Use MPI for inter-process communication (for HPC systems)

*********
 Examples
*********

Basic Coupled Run
=================

.. code-block:: bash

   anemoi-inference couple coupled_config.yaml

With Overrides
==============

.. code-block:: bash

   anemoi-inference couple coupled_config.yaml date=2025-01-15T00 lead_time=120

Using Defaults
==============

.. code-block:: bash

   anemoi-inference couple --defaults base_config.yaml coupled_config.yaml

.. seealso::

   - :ref:`run-command` - Run a single inference task
   - :ref:`config_introduction` - Configuration file reference
