.. _forcings:

##########
 Forcings
##########

:ref:`inputs` refers to the input methods used to fetch the initial
conditions. If the model need data during the run (dynamic forcings),
the forcings will be fetched by default from the same source as the
input. However, you can specify a different source.

The example below shows an example where the forcings are fetched from
``mars`` while the initial conditions are fetched from ``test``.

.. literalinclude:: forcings_1.yaml
