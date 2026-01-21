.. _sanitise-command:

#################
 Sanitise Command
#################

The ``sanitise`` command cleans and normalises checkpoint metadata to ensure
compatibility and remove sensitive or unnecessary information. This is
particularly useful when preparing checkpoints for sharing, publication, or
moving between different environments.

************
 Description
************

The ``sanitise`` command processes a checkpoint file and:

- Removes or anonymises sensitive metadata (e.g., internal paths, user information)
- Normalises metadata structure to current standards
- Ensures dataset source information is properly formatted
- Validates metadata integrity

This operation modifies the checkpoint file in-place. The command will only
make changes if necessary - if the metadata is already sanitised, the file
remains unchanged.

.. warning::

   This command modifies the checkpoint file in-place. It is recommended to
   create a backup before running this command.

.. note::

   The sanitisation process is idempotent - running it multiple times on the
   same checkpoint will not cause any issues.

.. argparse::
   :module: anemoi.inference.__main__
   :func: create_parser
   :prog: anemoi-inference
   :path: sanitise

*********
 Examples
*********

Basic Usage
===========

.. code-block:: bash

   anemoi-inference sanitise /path/to/checkpoint.ckpt

The command will output whether changes were made:

.. code-block:: text

   INFO Patching metadata

Or if no changes were needed:

.. code-block:: text

   INFO Metadata is already sanitised

Sanitise Before Sharing
========================

.. code-block:: bash

   # Create a copy first
   cp my_model.ckpt my_model_public.ckpt

   # Sanitise the copy
   anemoi-inference sanitise my_model_public.ckpt

*********************
 What Gets Sanitised
*********************

The sanitisation process handles various aspects of the metadata:

Paths and Locations
===================

- Internal file system paths are normalised or removed
- User home directories are anonymised
- Temporary paths are cleaned

Dataset Sources
===============

- Dataset source information is validated and normalised
- Ensures required fields are present
- Removes redundant or deprecated fields

User Information
================

- Removes or anonymises user-specific metadata
- Cleans training environment details that may contain sensitive info

Metadata Structure
==================

- Updates deprecated metadata formats to current standards
- Ensures compatibility with latest inference code
- Validates metadata schema

***********
 Use Cases
***********

Before Publishing Models
========================

When preparing to publish a model to a registry or share publicly, sanitise
the checkpoint to remove internal paths and sensitive information:

.. code-block:: bash

   anemoi-inference sanitise model.ckpt
   # Now safe to upload to Hugging Face, etc.

Moving Between Systems
======================

When transferring checkpoints between different systems (e.g., from training
cluster to inference server), sanitisation ensures paths are normalised:

.. code-block:: bash

   # On source system
   anemoi-inference sanitise checkpoint.ckpt
   scp checkpoint.ckpt destination:/path/

   # Checkpoint is ready to use on destination

Model Registry Preparation
==========================

Before registering a model in anemoi-registry, sanitise to ensure metadata
meets standards:

.. code-block:: bash

   anemoi-inference sanitise checkpoint.ckpt
   anemoi-registry register checkpoint.ckpt --name my-model-v1

Checkpoint Verification
========================

Use sanitise as part of a validation pipeline to ensure checkpoints meet
quality standards:

.. code-block:: bash

   # Sanitise
   anemoi-inference sanitise checkpoint.ckpt

   # Validate
   anemoi-inference validate checkpoint.ckpt

   # Inspect
   anemoi-inference inspect checkpoint.ckpt

******************
 Technical Details
******************

The sanitisation process:

1. Loads the checkpoint metadata and supporting arrays
2. Creates a deep copy of the metadata
3. Applies sanitisation rules from ``anemoi.utils.sanitise``
4. Compares sanitised metadata with original
5. If differences exist, replaces the metadata in the checkpoint file
6. Preserves all model weights and supporting arrays

The operation is safe because:

- Model weights are never modified
- Supporting arrays are preserved
- Only metadata is processed
- Original structure is maintained
- Changes are only applied if validation passes

.. seealso::

   - :ref:`inspect-command` - Inspect checkpoint metadata
   - :ref:`validate-command` - Validate checkpoint integrity
   - :ref:`patch-command` - Modify specific metadata fields
   - :ref:`metadata-command` - Extract checkpoint metadata
