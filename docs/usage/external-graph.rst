.. _usage-external-graph:

###################################
 Inference using an external graph
###################################

Anemoi is a framework for building and running machine learning models
based on graph neural networks (GNNs). One of the key features of such
GNNS is that they can operate on arbitrary graphs. In particular it
means one can train the model on one graph, but use it in inference on
another graph. This way one can transfer the model to a different domain
or dataset, without any fine tuning, or even change the scope of a
model. For example using a model trained as a stretched grid as a
limited area model (LAM) with boundary forcings in inference.

We should caution that such transfer of the model from one graph to
another is not guaranteed to lead to good results. Still, it is a
powerful tool to explore generalizability of the model or to test
performance before starting fine tuning through transfer learning.

The ability to do inference with an alternative graph, or more precisely
one 'external' to the checkpoint created in training, is supported by
anemoi-inference through the ``external_graph`` runner.

This runner, and the graph it will use, can be specified in the config
file as follows:

.. literalinclude:: yaml/external-graph1.yaml
   :language: yaml

In case one wants to run a model trained on a global dataset on a graph
supported only on a limited area one needs to specify the
``output_mask`` to be used. This mask selects the region on which the
model will forecast and triggers boundary forcings to be applied when
forecasting autoregressively towards later lead times. As in training,
also in inference the output mask orginates from an attribute of the
output nodes of the graph. It can be specified in the config file as
follows:

.. literalinclude:: yaml/external-graph2.yaml
   :language: yaml

For LAM models the limited area among the input nodes of a larger
dataset is often specified by the ``indices_connected_nodes`` attribute
of the input nodes. Anemoi-inference will automatically update the
dataloader to load only data in the limited area in case the external
graph contains this attribute and was build using the same dataset as
the one in the checkpoint.

In case one wants to work with a graph that was built on another dataset
than that used in training, on should specify this in the config file as
well:

.. literalinclude:: yaml/external-graph3.yaml
   :language: yaml

If you wish to run the external graph runner, but without the
anemoi-dataset configuration, some supporting arrays may need to be
updated. These can be sourced from either the ``graph['data']`` or from
a file on disk. This is done with the ``update_supporting_arrays`` key

.. literalinclude:: yaml/external-graph4.yaml
   :language: yaml

It should be emphasized that by using this runner the model will be
rebuilt and for this reason will differ from the model stored in the
checkpoint. To avoid unexpected results, there is a default check that
ensures the model used in inference has the same weights, biases and
normalizer values as that stored in the checkpoint. In case of a more
adventurous use-case this check can be disabled through the config as:

.. literalinclude:: yaml/external-graph5.yaml
   :language: yaml
