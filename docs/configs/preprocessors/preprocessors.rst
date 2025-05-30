.. _preprocessors:

###############
 Preprocessors
###############

In addition to the :ref:`inputs`, you can also specify preprocessors to
modify the input data before it is used by the model. The preprocessors
are applied in the order they are specified in the configuration.

*********
 Overlay
*********

While this preprocessor has no meaningful impact on properly running a
model, it can be used to test a models ability to cope with out of
distribution features within an input field.

It works by overlaying an image or array onto a field, with the option
to rescale the values.

.. literalinclude:: overlay_1.yaml
   :language: yaml
