.. _dev-guide:

#########################
 Technical documentation
#########################

This set of pages describes the package from the point of view of a
technical contributor, rather than a user, to anemoi-inference.

If you are interested in extending anemoi-inference or contributing back
into the package, welcome! Start with our :ref:`contributor's guide
<dev-contributing>`.

*******************
 Codebase overview
*******************

`anemoi-inference` is built on top of a `registry pattern
<https://realpython.com/ref/computer-science-glossary/registry-pattern/>`_.
The structure of the code follows the structure of the configuration
files.

Each registry-backed component has a set of classes defined under
``src/anemoi/inference/<component_name>/``. Each class inherits from the
base class ``src/anemoi/inference/<component_name>.py``. For example,
one can define ``plot`` as an output. The code which runs the ``plot``
output is found in ``src/anemoi/inference/outputs/plot.py``.

The baseclass ``Output`` in ``src/anemoi/inference/output.py`` defines
all shared or default behavior, which is then overridden as needed by
the subclasses.

How is the correct subclass found and described? Each subclass uses the
``@<component_name>_registry.register("<name>")`` decorator to register
itself into the central registry. Each component has its own registry,
which is defined under
``src/anemoi/inference/<component_name>/__init__.py``.

So, to indicate the the ``plot`` output should be registered to the
``output_registry``, it is tagged with
``@output_registry.register("plot")``. When an end user configures the
yaml to use a ``plot`` output, the runner retrieves the matching class
from the registry and returns it.

.. code:: yaml

   output:
     plot:
       dir: test_dir/

If the above configuration snippet is passed in, the class registered
under ``"plot"`` (in this case, ``PlotOutput``) is instantiated, with
any configuration included as a lower layer passed into the matching
argument on the class (in this case, it would be instantiated as
``PlotOutput(..., dir="test_dir/")``). More on what you can expect from
configuration later.

These are all the registry-backed components that follow this pattern:

-  **Inputs**: ``input.py``, ``src/anemoi/inference/inputs/``

-  **Outputs**: ``output.py``, ``src/anemoi/inference/outputs``

-  **Runners**: ``runner.py``, ``src/anemoi/inference/runners``

-  **Transports**: ``transport.py``, ``src/anemoi/inference/transports``

-  **Processors**: ``processor.py``. Note Processor is a special case,
   with mid-, pre-, and post- processors having their own registries but
   a shared base class.

For more information on how to add a new option to an existing registry,
or update an existing option, check out
:ref:`dev-add-new-configuration-option`.

******************
 RunConfiguration
******************

RunConfiguration is a pydantic class which describes at a high level the
expected configuration formatting.
