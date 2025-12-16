.. _usage-advanced-remote:

####################
 Remote Checkpoints
####################

It is possible to run with a checkpoint stored in huggingface directly
by specifying the checkpoint as follows:

.. literalinclude:: yaml/remote1.yaml
   :language: yaml

.. warning::

   To use huggingface stored models requires `huggingface_hub
   <https://github.com/huggingface/huggingface_hub>`_ to be installed in
   your environment.

   .. code:: bash

      pip install huggingface_hub
