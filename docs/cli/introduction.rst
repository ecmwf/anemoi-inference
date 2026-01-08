.. _cli-commands:

Command Line Tool
==================

When you install the `anemoi-inference` package, this will also install command line tool
called ``anemoi-inference`` this can be used to run inference and manage checkpoints.

The tools can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-inference --help


The commands are organised into two categories:

Inference Commands
------------------

Commands for running model inference:

- :ref:`Run Command <run-command>` - Run inference with a trained model
- :ref:`Couple Command <couple-command>` - Run coupled inference tasks
- :ref:`Retrieve Command <retrieve-command>` - Generate data retrieval requests

Checkpoint Management Commands
------------------------------

Commands for inspecting and managing checkpoint files:

- :ref:`Metadata Command <metadata-command>` - Extract checkpoint metadata
- :ref:`Inspect Command <inspect-command>` - Inspect checkpoint contents
- :ref:`Validate Command <validate-command>` - Validate checkpoint integrity
- :ref:`Patch Command <patch-command>` - Modify checkpoint metadata
- :ref:`Sanitise Command <sanitise-command>` - Clean and normalise checkpoint metadata
- :ref:`Requests Command <requests-command>` - Show MARS requests for a configuration
