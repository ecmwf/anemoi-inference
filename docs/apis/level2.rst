.. _api_level2:

#####################
 Object oriented API
#####################

The object oriented API is more flexible than the NumPy API, as it
allows users to use classes that can create the initial state
(:class:`Input <anemoi.inference.inputs.Input>`), and classes that can
process the output state (:class:`Output
<anemoi.inference.outputs.Output>`). Several classes are provided as
part of the package, and users can create their own classes to support
various sources of data and output formats.

.. literalinclude:: code/level2.py
   :language: python
