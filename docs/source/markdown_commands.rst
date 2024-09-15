.. _markdown_commands:

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

Markdown Commands
=================

`nbrefactor` allows the use of Markdown Commands to control the refactoring process. 
These commands allow you to manipulate the structure of the refactored modules, or to ignore certain parts of the notebook.

Basic Usage
-----------

To use a Markdown command, you may include it in an HTML comment (as to not interfere with your Notebook's aesthetics 😇) within a Markdown cell. 

The format is:

.. code-block:: markdown

    <!-- $command-name=value -->

For example, to rename a package:

.. code-block:: markdown

    <!-- $package=my_custom_package -->

Markdown Command Table
----------------------

The following table lists all the possible Markdown commands and their functions.

.. table::
   :widths: 30 70

   ==========================================  ============================================================
   Command                                      Description
   ==========================================  ============================================================
   ``$ignore-package``                          Ignores all modules/packages until a header with a depth 
                                                less than or equal to the current one is reached.
   ``$ignore-module``                           Ignores a single module (may consist of multiple code cells).
   ``$ignore-cell``                             Ignores the next code cell regardless of type.
   ``$ignore-markdown``                         Ignores the current Markdown cell (e.g., when used for 
                                                instructions only).
   ``$package=<name>``                          Renames the current package and asserts the node type as 
                                                'package'.
   ``$module=<name>``                           Renames the current module and asserts the node type as 
                                                'module'.
   ``$node=<name>``                             Renames the current node generically, regardless of type.
   ``$declare-package=<name>``                  Declares a new node and asserts its type as 'package'.
   ``$declare-module=<name>``                   Declares a new node and asserts its type as 'module'.
   ``$declare-node=<name>``                     Declares a new node with no type (type will be inferred).
   ==========================================  ============================================================

Command Usage Examples
----------------------

### Ignore Package

This command will ignore all packages or modules under the current header level until a smaller or equal depth header is encountered.

.. code-block:: markdown

    <!-- $ignore-package -->

### Ignore Module

Ignores a single module, skipping all code cells that belong to it.

.. code-block:: markdown

    <!-- $ignore-module -->

### Ignore Cell

Ignore the next code cell, regardless of its type. This is useful when a specific code cell should not be included in the refactored module.

.. code-block:: markdown

    <!-- $ignore-cell -->

### Rename a Package

To rename the current package, use the `package` command. For example:

.. code-block:: markdown

    <!-- $package=my_custom_package -->

This command will rename the current package to `my_custom_package`.

### Declare a Module

The `declare-module` command creates a new module node. For example:

.. code-block:: markdown

    <!-- $declare-module=new_module -->

This will declare a new module named `new_module`.

### Rename a Node

The `node` command renames the current node, regardless of its type. Use it as follows:

.. code-block:: markdown

    <!-- $node=my_custom_node -->

This will rename the current node to `my_custom_node`.

### Declare a Generic Node

Declare a new node that doesn't have a type yet, and let the refactoring process infer its type:

.. code-block:: markdown

    <!-- $declare-node=my_generic_node -->

This declares a new node named `my_generic_node`, and its type will be inferred during the refactoring process.
