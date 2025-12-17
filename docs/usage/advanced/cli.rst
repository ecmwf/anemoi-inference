.. _usage-advanced-cli:

###########################
 Advanced Command line Use
###########################

You can run the inference from the command line using the
:ref:`anemoi-inference run <run-command>` command.

You must first create a configuration file in YAML format. The simplest
configuration must contain the path to the checkpoint:

.. literalinclude:: yaml/cli1.yaml
   :language: yaml

Then you can run the inference with the following command:

.. literalinclude:: code/cli1.sh
   :language: bash

The other entries in the configuration file are optional, and will be
substituted by the default values if not provided.

You can also override values by providing them on the command line:

.. literalinclude:: code/cli2.sh
   :language: bash

Overrides are parsed as an `OmegaConf
<https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#from-a-dot-list>`_
dotlist, so list items can be accessed with ``list.index`` or
``list[index]``.

You can also run entirely from the command line without a config file,
by passing all required options as an override:

.. literalinclude:: code/cli3.sh
   :language: bash

The configuration below shows how to run the inference from the data
that was used to train the model, by setting ``dataset`` entry to
``true``:

.. literalinclude:: yaml/cli2.yaml
   :language: yaml

Below is an example of how to override list entries and append to lists
on the command line by using the dotlist notation. Running inference
with following command:

.. literalinclude:: code/cli4.sh
   :language: bash

together with configuration file:

.. literalinclude:: yaml/cli3.yaml
   :language: yaml

will overide the first entry in the ``input.dataset.cutout`` list with
the dictionary ``{"dataset": "./analysis_20240131_00.zarr"}`` and will
append the dictionary ``{"dataset": "./lbc_20240131_00.zarr"}`` to it.

The configuration below shows how to provide run the inference for a
checkpoint that was trained with one the ICON grid:

.. literalinclude:: yaml/cli4.yaml
   :language: yaml

See :ref:`run-command` for more details on the configuration file.

.. warning::

   This is still work in progress, and content of the YAML configuration
   files will change and the examples above may not work in the future.

.. seealso::

   - :ref:`run-command` - Run command documentation
   - :ref:`config_introduction` - Configuration file reference
   - :ref:`usage-quickstart` - Quickstart guide
   - :ref:`usage-advanced-sources` - Input data sources
   - :ref:`usage-advanced-saving` - Output configuration
