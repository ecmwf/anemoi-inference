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
for example one of the :ref:`inputs <inputs>`, the only code that needs
to be updated is the single affected class matching the input.

So, if you wanted to add a new option to the repeated-dates input, it
only requires updates to :class:`RepeatedDatesInput
<anemoi.inference.inputs.repeated_dates.RepeatedDatesInput>`.

Adding a new option
===================

Let's look at a more specific example with the repeated-dates input.
Let's say the current input can only accept its own existing options,
plus the default configuration options (as defined in the abstract base
class :class:`Input <anemoi.inference.input.Input>`).

.. code:: yaml

   input:
     repeated-dates:
       source: mars
       date: 2024-01-01

When the ``RepeatedDatesInput`` class receives this configuration, it
matches with the keyword argument on the class:

.. code:: python

   input = RepeatedDatesInput(source='mars', date='2024-01-01')

Note: due to the registry pattern, this is not written anywhere in the
code, but instead built using ``inputs/__init__.py::create_input()``
(see :ref:`modules-inputs`).

In order to add a new configuration option, simply add a new keyword
argument to the class:

.. code:: python

   @input_registry.register("repeated-dates")
   class RepeatedDatesInput(Input):
       """This class is identical to the one used to in anemoi-datasets/create
       It uses a source of constants (e.g. a source containing the bathymetry)
       available only for a given date and returns its content whever date
       is requested by the runner
       """

       trace_name = "repeated dates"

       def __init__(
           self,
           context: Context,
           metadata: Metadata,
           *,
           source: str,
           mode: str = "constant",
           new_variable: str | None = None,
           **kwargs: Any,
       ) -> None:
           super().__init__(
               context=context,
               ...
           )
           self.new_variable = new_variable

Notice the type-hint, which follows our :ref:`style guide
<style-guide>`. As this is user-inputted, make sure to validate the
input and throw an error if it isn't what you're expecting. Add the new
variable to the docstring of the class and user documentation.

.. code:: yaml

   input:
     repeated-dates:
       source: mars
       date: 2024-01-01
       new_variable: "hello world"


******************************
 Updating every sub-component
******************************

Similarly, if this is shared behavior, you can instead update the shared
superclass (in this case, :class:`Input <anemoi.inference.input.Input>`)

.. code:: python

   class Input(ABC):
       """Abstract base class for input handling."""

       def __init__(
           self,
           context: "Context",
           metadata: "Metadata",
           *,
           variables: list[str] | None = None,
           pre_processors: list[ProcessorConfig] | None = None,
           purpose: str | None = None,
           new_variable: str | None = None,
       ) -> None:
           self.new_variable = new_variable
           ...

In this case, ensure you also update every subclass to also pass the
variable up:

.. code:: python

   class RepeatedDatesInput(Input):
       def __init__(
           self,
           context: Context,
           metadata: Metadata,
           *,
           source: str,
           mode: str = "constant",
           new_variable: str | None = None,
           **kwargs: Any,
       ) -> None:
           super().__init__(
               context=context,
               ...,
               new_variable=new_variable,
           )

If a subclass has specific handling you want to implement to override
the default behavior, you can also update that in specific subclasses.
Try to reduce repetition in code using shared methods in the superclass
when you can.

****************************
 Adding a new sub-component
****************************

A new sub-component should: inherit from an existing base class, include
all existing variables on the base class, and include a registry entry
corresponding to the name.

.. code:: python

   @input_registry.register("new-input")
   class NewInput(Input):
       """This is an example new input."""

       def __init__(
           self,
           context: "Context",
           metadata: "Metadata",
           *,
           variables: list[str] | None = None,
           pre_processors: list[ProcessorConfig] | None = None,
           purpose: str | None = None,
       ) -> None:
           super().__init__(
               context=context,
               ...
           )

Ensure this is well-documented, including in the user documentation.
