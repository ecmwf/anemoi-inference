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

.. literalinclude:: yaml/forcings_1.yaml
   :language: yaml

This example above is a shortcut for:

.. literalinclude:: yaml/forcings_2.yaml

You can also specify different sources for each forcing.

This is to get the initial constant forcings from a file:

.. literalinclude:: yaml/forcings_3.yaml
   :language: yaml

Get the dynamic forcings from mars:

.. literalinclude:: yaml/forcings_4.yaml
   :language: yaml

And the LAM boundary conditions from a mars as well:

.. literalinclude:: yaml/forcings_5.yaml
   :language: yaml

In the case of the last three type of forcings, it they are not
specified, the value of `forcings.input` will be used. It this value is
not specified, the value of `input` will then be used.
