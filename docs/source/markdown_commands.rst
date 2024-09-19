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

.. code-block:: html

    <!-- $command-name=value -->

For example, to rename a package:

.. code-block:: html

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

