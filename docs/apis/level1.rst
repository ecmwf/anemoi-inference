.. _api_level1:

####################
 NumPy to NumPy API
####################

The simplest way to run a inference from a checkpoint is to provide the
initial state as a dictionary containing NumPy arrays for each input
variable.

You then create a Runner object and call the `run` method, which will
yield the state at each time step. Below is a simple code example to
illustrate this:

.. literalinclude:: code/state.py
   :language: python
