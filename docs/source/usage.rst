.. _usage:

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

Command Line Interface
======================

`nbrefactor` provides a CLI to easily refactor notebooks into a structured project hierarchy.

Basic CLI Usage
---------------

To use the CLI, run the following command:

.. code-block:: bash

    jupyter nbrefactor <notebook_path> <output_path> [OPTIONS]

- ``notebook_path``: Path to the Jupyter notebook file you want to refactor.
- ``output_path``: Directory where the refactored Python modules will be saved.

CLI Arguments
-------------

.. rst-class:: arguments-cli-table

.. table::
   :widths: 25 45 30

   ==================================  ==========================================  ===============
       Argument                             Type                                      Default
   ==================================  ==========================================  ===============
   ``notebook_path``                        `str`                                       (Required)
   ``output_path``                          `str`                                       (Required)
   ``-rp``, ``--root-package``              `str`                                       ``"."``
   ``-gp``, ``--generate-plot``             `flag`                                      ``False``
   ``-pf``, ``--plot-format``               `str`                                       ``"pdf"``
   ==================================  ==========================================  ===============

Example Usages
--------------

1. **Basic Refactor**: Refactor a notebook and save the output in a specified directory:

   .. code-block:: bash

      jupyter nbrefactor notebook.ipynb output_dir

2. **Specify a Root Package**: Refactor and assign a root package name:

   .. code-block:: bash

      jupyter nbrefactor notebook.ipynb output_dir --root-package my_package

3. **Generate a Plot**: Refactor and generate a plot of the module hierarchy:

   .. code-block:: bash

      jupyter nbrefactor notebook.ipynb output_dir --generate-plot

4. **Generate a Plot with Specific Format**: Generate a plot in PNG format:

   .. code-block:: bash

      jupyter nbrefactor notebook.ipynb output_dir --generate-plot --plot-format png
