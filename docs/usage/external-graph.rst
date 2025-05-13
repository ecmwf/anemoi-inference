.. _usage-getting-started:

#################################
Inference using an external graph
#################################

Anemoi is a framework for building and running machine learning models based on graph neural networks (GNNs).
One of the key features of such GNNS is that they can operate on arbitrary graphs. In particular it means one can train
the model on one graph, but use it in inference on another graph. This way one can transfer the model to a different domain
or dataset, without any fine tuning, or even change the scope of a model. For example using a model trained as a stretched grid as a
limited area model (LAM) with boundary forcings in inference.

We should caution that such transfer of the model from one graph to another is not guaranteed to lead to good results. Still, it is a powerful
tool to explore generalizability of the model or to test performance before starting fine tuning through transfer learning.

The ability to do inference with an alternative graph, or more precisely one 'external' to the checkpoint created in training, is supported by
anemoi-inference through the `external_graph` runner.

This runner, and the graph it will use, can be specified in the config file as follows:
.. literalinclude:: yaml/external_graph1.yaml
   :language: yaml

In case one wants to run a model trained on a global dataset on a graph supported only on a limited area one needs to specify the `output_mask` to be used.
This mask selects the region on which the model will forecast and triggers boundary forcings to be applied when forecasting autoregressively towards later lead times.
As in training, also in inference the output mask orginates from an attribute of the output nodes of the graph. It can be specified in the config file as follows:
.. literalinclude:: yaml/external_graph2.yaml
   :language: yaml

For LAM models the limited area among the input nodes of a larger dataset is often specified by the `indices_connected_nodes` attribute of the input nodes.
Anemoi-inference will automatically update the dataloader to load only data in the limited area in case the external graph was build using the same dataset as the one in the checkpoint.
