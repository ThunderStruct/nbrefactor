<p align="center">
    <img src="https://i.imgur.com/ukBP39X.png" alt="nbrefactor Logo" width="420">
</p>

<br />

<div align="center">

<a href="https://github.com/ThunderStruct/nbrefactor">![Platform](https://img.shields.io/badge/python-v3.7-green)</a>
<a href="https://pypi.org/project/nbrefactor/">![pypi](https://img.shields.io/badge/pypi%20package-0.1.6-lightgrey.svg)</a>
<a href="https://github.com/ThunderStruct/nbrefactor/blob/master/LICENSE">![License](https://img.shields.io/badge/license-MIT-orange)</a>
<a href="https://nbrefactor.readthedocs.io/en/latest/">![Read the Docs](https://readthedocs.org/projects/nbrefactor/badge/?version=latest)</a>
<a href="https://github.com/ThunderStruct/nbrefactor/actions/workflows/ci.yml">![GitHub CI](https://github.com/ThunderStruct/nbrefactor/actions/workflows/ci.yml/badge.svg)</a>



</div>

<p align="center">
An automation tool to refactor Jupyter Notebooks to Python packages and modules.
</p>

---

# Overview

**nbrefactor** is designed to refactor Jupyter Notebooks into structured Python packages and modules. Using Markdown Headers and/or custom commands in a notebook's Markdown/text cells, nbrefactor creates a hierarchical module structure that reflects the notebook's content autonomously.

## Motivation

With the growing dependence on cloud-based IPython platforms ([Google Colab](https://colab.research.google.com/), primarily), developing projects directly in-browser has become more prominent. Having suffered through the pain of refactoring entire projects from Jupyter Notebooks into Python packages/modules to facilitate PyPI publication (and proper source control), this tool was developed to automate the refactoring process.

# Implementation

This project does _not_ just create a hierarchy based on the level of Markdown headers (how many `#` there are); this is just a single step in the refactoring process.

Since we are generating modules that potentially depend on context from previous cells in the notebook, dependency-analysis is required. Furthermore, we also need track the generated modules and all globally-accessible identifiers throughout the notebook to generate relative import statements as needed.

For instance, if a class is refactored to a generated module `./package/sub_package/module.py`, this definition and module path need to be tracked so we can relatively import it as needed if it appears in successive cells or modules. _Scope-Awareness_ and _Identifier-Shadowing_ posed a challenge as well, and are also handled in the dependency analysis phase of the refactoring.

## Module Hierarchy Generation

Convert markdown headers in notebooks into a corresponding folder and file structure.

![refactoring_examples](https://i.imgur.com/bBgHJay.png)

## Code Dependency Analyzer (CDA)

The core of **nbrefactor**'s functionality lies in the Code Dependency Analyzer (CDA). The CDA is responsible for parsing code cells, tracking declared definitions, and analyzing dependencies across the generated modules. This module tackles challenges that were raised during the inception of the refactoring-automation process (primarily handling relative imports dynamically as we generate the modules, identifier shadowing, and non-redundant dependency injection).

1. **IPython Magic Command Removal**: clean the source code by omitting IPython magic commands (to ensure that the code can be parsed by Python's `ast`).
2. **AST Parsing**: parse the sanitized code into an Abstract Syntax Tree
3. **Import Statement Stripping**: extract and strip import statements from the parsed code, and add them to a global (across all cells) tracker.
4. **Global Definition Tracking**: track all encountered definitions (declared functions and classes) globally. This inherently handles identifier shadowing.
5. **Dependency Analysis**: analyze identifier usages in a given code block.
6. **Dynamic Relative Import Resolution**: resolve local import statements dynamically depending on the current and target modules' positions in the tree.
7. **Dependency Generation and Resolution**: generate the respective import statements (given the definitions' analysis in step 5 & 6) to be injected during the file-writing phase.

# Installation

## PyPI (recommended)

The Python package is hosted on the [Python Package Index (PyPI)](https://pypi.org/project/nbrefactor/).

The latest published version of **nbrefactor** can be installed using

```bash
pip install nbrefactor
```

## Manual Installation

Simply clone the repo and extract the files in the `nbrefactor` folder,
then run:

```bash
pip install -r requirements.txt
pip install -e .
```

Or use one of the scripts below:

### GIT

- `cd` into your project directory
- Use `sparse-checkout` to pull the library files only into your project directory
  ```bash
  git init nbrefactor
  cd nbrefactor
  git remote add -f origin https://github.com/ThunderStruct/nbrefactor.git
  git config core.sparseCheckout true
  echo "nbrefactor/*" >> .git/info/sparse-checkout
  git pull --depth=1 origin master
  pip install -r requirements.txt
  pip install -e .
  ```

### SVN

- `cd` into your project directory
- `checkout` the library files
  ```bash
  svn checkout https://github.com/ThunderStruct/nbrefactor/trunk/nbrefactor
  pip install -r requirements.txt
  pip install -e .
  ```

# Usage

Refer to the [documentation](https://nbrefactor.readthedocs.io/en/latest/) for the comprehensive commands' reference. Some basic usages are provided below.

Sample basic usage:
```bash
jupyter nbrefactor notebook.ipynb output_dir/
```

## Command Line Interface

`nbrefactor` provides a CLI to easily refactor notebooks into a structured project hierarchy.

### Basic CLI Usage

To use the CLI, run the following command:

```bash
jupyter nbrefactor <notebook_path> <output_path> [OPTIONS]
```

| Options | Description                                        |
|---------|----------------------------------------------------|
| -i      | Generate __init__.py files in package directories. |
| -rp     | Name of the root package (defaults to ".")         |
| -gp     | Generate a plot of the module dependency tree      |
| -pf     | Format of the plot (e.g., "pdf", "png")            |

- `<notebook_path>`: Path to the Jupyter notebook file you want to refactor.
- `<output_path>`: Directory where the refactored Python modules will be saved.


## Markdown Commands

The following table lists the currently implemented Markdown commands and their functions.

| Command                         | Description                                                                                                |
|---------------------------------|------------------------------------------------------------------------------------------------------------|
| `$ignore-package`               | Ignores all modules/packages until a header with a depth less than or equal to the current one is reached. |
| `$ignore-module`                | Ignores a single module (may consist of multiple code cells).                                              |
| `$ignore-cell`                  | Ignores the next code cell regardless of type.                                                             |
| `$ignore-markdown`              | Ignores the current Markdown cell (e.g., when used for instructions only).                                 |
| `$package=<name>`               | Renames the current package and asserts the node type as 'package'.                                        |
| `$module=<name>`                | Renames the current module and asserts the node type as 'module'.                                          |
| `$node=<name>`                  | Renames the current node generically, regardless of type.                                                  |
| `$declare-package=<name>`       | Declares a new node and asserts its type as 'package'.                                                     |
| `$declare-module=<name>`        | Declares a new node and asserts its type as 'module'.                                                      |
| `$declare-node=<name>`          | Declares a new node with no type (type will be inferred).                                                  |


## Demo

There are several example notebooks provided to showcase **nbrefactor**'s capabilities.

- [_Primary Demo Notebook_](src/demo/examples/sample_primary_demo.ipynb): this notebook contains several examples of the core nbrefactor features, including all Markdown commands.
- [_CS231n Notebook_](src/demo/examples/sample_CS231n_colab.ipynb): the official CS231n Colab notebook.
- [_HiveNAS Notebook_](src/demo/examples/sample_HiveNAS.ipynb): a larger project with a more complex folder structure.
- [_Markdown-only Notebook_](src/demo/examples/sample_markdown_only.ipynb): a Markdown-only notebook to illustrate the directory-refactoring abilities of nbrefactor.

### Interactive Demo

An interactive Notebook-based demo can be found [here](src/demo/demo.ipynb), which can be used to run the example projects discussed above.

# Change Log

Consult the [CHANGELOG](CHANGELOG.md) for the latest updates.

# Contributing

All contributions are welcome (and encouraged)! Even incremental PRs that just add minor features or corrections to the docs will be considered :) 

If you'd like to contribute to **nbrefactor**, please read the [CONTRIBUTING](CONTRIBUTING.md) guidelines. 

The [TODO](TODO.md) list delineates some potential future implementations and improvements. 

## PR Submission

In addition to following the [contribution guidelines](CONTRIBUTING.md), please ensure the steps below are adhered to prior to submitting a PR:

- The [CHANGELOG](CHANGELOG.md) is updated according to the given structure
- The [README](README.md) and [TODO](TODO.md) are updated (if applicable)

# License

**nbrefactor** is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
