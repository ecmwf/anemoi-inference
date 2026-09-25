.. _dev-add-new-configuration-option:

#################################
 Adding to Configuration options
#################################

As a developer, perhaps you are looking to add or update an option on an
existing registry-backed component. Or, you instead want to add a new
input, output, or processor. You're in the right place!

With the registry pattern used inside the repo, adding to or updating
these components does not require extensive code-base changes.

*********************************
 Updating a single sub-component
*********************************

If your feature is only needed for one specific configuration setting,
for example one of the :ref:`outputs <outputs>`, the only code that
needs to be updated is the single affected class matching the output.

So, if you wanted to add a new option to the NetCDF output, it only
requires updates to :class:`NetCDFOutput
<anemoi.inference.outputs.netcdf.NetCDFOutput>`.

Class overview
==============

Each of the classes has a couple of key features you will need to be
aware of.

.. code:: python

   @output_registry.register("netcdf")
   class NetCDFOutput(Output):
       """NetCDF output class."""

       def __init__(
           self,
           context: Context,
           metadata: Metadata,
           *,
           path: Path,
           **kwargs: Any,
       ) -> None:
           super().__init__(
               context=context,
               metadata=metadata,
               path=path,
           )

#. `@output_registry.register("netcdf")` -> this line registers the
   subclass into the registry, under the name "netcdf".

#. `class NetCDFOutput(Output):` -> The class inherits from the `Output`
   base class.

#. `context` and `metadata` variables -> These are handled by the runner
   class, and automatically included. They do not originate directly
   from the config.

#. `*` -> This is a piece of Python syntax, indicating that variables
   should always be defined with their full names (ie
   `NetCDFOutput(path='path')` rather than `NetCDFOutput('path')`)

#. `path: Path` -> these are arguments which come directly from the
   configuration. If it is optional, it should be a keyword with a
   useful default or `None`.

#. `super().__init__(...)` -> this line passes up these variables to the
   `__init__` function of the `Output` base class.

Adding a new option
===================

Let's look at a more specific example with the NetCDF output. Let's say
the current output can only accept its own existing options, plus the
default configuration options (as defined in the abstract base class
:class:`Output <anemoi.inference.output.Output>`).

.. code:: yaml

   output:
     netcdf:
       path: output.nc
       float_size: f4

When the ``NetCDFOutput`` class receives this configuration, it matches
with the keyword argument on the class:

.. code:: python

   output = NetCDFOutput(path='output.nc', float_size='f4')

Note: due to the registry pattern, this is not written anywhere in the
code, but instead built using ``outputs/__init__.py::create_output()``
(see :ref:`modules-outputs`).

In order to add a new configuration option, simply add a new keyword
argument to the class:

.. code:: python

   @output_registry.register("netcdf")
   class NetCDFOutput(Output):
       """NetCDF output class."""

       def __init__(
           self,
           context: Context,
           metadata: Metadata,
           *,
           path: Path,
           float_size: str = "f4",
           missing_value: float | None = np.nan,
           new_variable: str | None = None,
           **kwargs: Any,
       ) -> None:
           super().__init__(
               context=context,
               ...
           )
           self.new_variable = new_variable

Notice the type-hint, which follows our :ref:`contributing guide
<dev-contributing>`. As this is user-inputted, make sure to validate the
input and throw an error if it isn't what you're expecting. Add the new
variable to the docstring of the class and user documentation.

.. code:: yaml

   output:
     netcdf:
       path: output.nc
       float_size: f4
       new_variable: "hello world"

******************************
 Updating every sub-component
******************************

Similarly, if this is shared behaviour, you can instead update the
shared superclass (in this case, :class:`Output
<anemoi.inference.output.Output>`)

.. code:: python

   class Output(ABC):
       """Abstract base class for output handling."""

       def __init__(
           self,
           context: "Context",
           metadata: "Metadata",
           *,
           variables: list[str] | None = None,
           post_processors: list[ProcessorConfig] | None = None,
           output_frequency: int | None = None,
           write_initial_state: bool | None = None,
           new_variable: str | None = None,
       ) -> None:
           self.new_variable = new_variable
           ...

In this case, ensure you also update every subclass to also pass the
variable up:

.. code:: python

   class NetCDFOutput(Output):
       def __init__(
           self,
           context: Context,
           metadata: Metadata,
           *,
           path: Path,
           float_size: str = "f4",
           new_variable: str | None = None,
           **kwargs: Any,
       ) -> None:
           super().__init__(
               context=context,
               ...,
               new_variable=new_variable,
           )

If a subclass has specific handling you want to implement to override
the default behaviour, you can also update that in specific subclasses.
Try to reduce repetition in code using shared methods in the superclass
when you can.

****************************
 Adding a new sub-component
****************************

If you would like to add a new sub-component, start with `plugins
<https://anemoi.readthedocs.io/projects/plugins/en/latest/guide/introduction.html>`_.
New sub-components should only be added to anemoi-inference if they are
useful for a wide variety of users. If it is for a specific application,
then creating a plugin (which can then be shared in a new repo if
needed) allows for flexibility around your requirements.

A new sub-component should: inherit from an existing base class, include
all existing variables on the base class, and include a registry entry
corresponding to the name.

.. code:: python

   @output_registry.register("new-output")
   class NewOutput(Output):
       """This is an example new output."""

       def __init__(
           self,
           context: "Context",
           metadata: "Metadata",
           *,
           variables: list[str] | None = None,
           post_processors: list[ProcessorConfig] | None = None,
           output_frequency: int | None = None,
           write_initial_state: bool | None = None,
       ) -> None:
           super().__init__(
               context=context,
               ...
           )

Ensure this is well-documented, including in the user documentation.
