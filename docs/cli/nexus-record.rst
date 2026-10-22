.. _nexus-record-command:

Nexus-record Command
====================

The ``nexus-record`` command prints a checkpoint's record for Anemoi Nexus:
its ``uuid`` and, as ``metadata``, the metadata embedded in the checkpoint with
``size`` (the file size in bytes) added. ``nexus-client`` sends it as is, and
never reads the checkpoint itself. Nexus names a model by its uuid unless
``--name`` gives one.

.. code:: console

   $ anemoi-inference nexus-record model.ckpt @attributes.yaml -o record.json
   $ nexus-client create models $(jq -r .uuid record.json) --file record.json

The record attributes Nexus needs to file the asset are given as options
(``--owner``, ``--project``/``--projects``, ``--license``/``--licenses``,
``--name``) or in an ``@FILE`` (JSON or YAML), in which ``project``/``projects``
and ``license``/``licenses`` may be singular or plural; the options win over
the file:

.. code:: yaml

   # attributes.yaml
   owner: alice
   project: MLP
   licenses: [CC-BY-4.0]

The attributes are handled by :mod:`anemoi.utils.nexus`, shared with
``anemoi-datasets nexus-record``.

.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :path: nexus-record
