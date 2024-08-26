
# PyNBRefactor



# ![PyNBRefactor Logo]())

 [![Platform](https://img.shields.io/badge/python-v3.7-green)](https://github.com/ThunderStruct/PyNBRefactor) [![pypi](https://img.shields.io/badge/pypi%20package-0.1.5-lightgrey.svg)](https://pypi.org/project/PyNBRefactor/0.1.0/) [![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/ThunderStruct/PyNBRefactor/blob/master/LICENSE) [![Read the Docs](https://readthedocs.org/projects/pynbrefactor/badge/?version=latest)](https://pynbrefactor.readthedocs.io/en/latest/)

An automation tool to refactor Jupyter Notebooks to Python packages and modules.

------------------------


## Overview (The "What")

PyNBRefactor is designed to convert Jupyter Notebooks into structured Python packages and modules. Using Markdown Headers and/or custom commands in a notebook's Markdown/text cells, PyNBRefactor creates a hierarchical module structure that reflects the notebook's content autonomously.

## Motivation (The "Why")

With the growing dependence on cloud-based IPython platforms ([Google Colab](https://colab.research.google.com/), primarily), developing projects directly in-browser has become more prominent. Having suffered through the pain of refactoring entire projects from Jupyter Notebooks into Python packages/modules to facilitate PyPI publication (and proper source control), this tool was developed to automate the refactoring process.

## Approach (The "How")

This project does *not* just create a hierarchy based on the level of Markdown headers (how many `#` there are); this is just a single step in the refactoring process.

Since we are generating modules that potentially depend on context from previous cells in the notebook, dependency-analysis is required. Furthermore, we also need track the generated modules and all globally-accessible identifiers throughout the notebook to generate relative import statements as needed. 

For instance, if a class is refactored to a generated module `./package/sub_package/module.py`, this definition and module path need to be tracked so we can relatively import it as needed if it appears in successive cells or modules. _Scope-Awareness_ and _Identifier-Shadowing_ posed a challenge as well, and are also handled in the dependency analysis phase of the refactoring.

### Module Hierarchy Generation

Convert markdown headers in notebooks into a corresponding folder and file structure.
<!-- finalize the example file and add an image here for a side-by-side comparison (before and after sort of thing) -->


### Code Dependency Analyzer (CDA)

The core of PyNBRefactor's functionality lies in the Code Dependency Analyzer (CDA). The CDA is responsible for parsing code cells, tracking declared definitions, and analyzing dependencies across the generated modules. This module tackles challenges that were raised during the inception of the refactoring-automation process (primarily handling relative imports dynamically as we generate the modules, identifier shadowing, and non-redundant dependency injection).

1. **IPython Magic Command Removal**: clean the source code by omitting IPython magic commands (to ensure that the code can be parsed by Python's `ast`).
2. **AST Parsing**: parse the sanitized code into an Abstract Syntax Tree
3. **Import Statement Stripping**: extract and strip import statements from the parsed code, and add them to a global (across all cells) tracker.
4. **Global Definition Tracking**: track all encountered definitions (declared functions and classes) globally. This inherently handles identifier shadowing.
5. **Dependency Analysis**: analyze identifier usages in a given code block.
6. **Dynamic Relative Import Resolution**: resolve local import statements dynamically depending on the current and target modules' positions in the tree.
7. **Dependency Generation and Resolution**: generate the respective import statements (given the definitions' analysis in step 5 & 6) to be injected during the file-writing phase.


## Installation

### PyPI (recommended)

### SVN Checkout


## Usage

Refer to the [documentation](readthedocslinkhere) for all CLI commands. Some common usages are provided below.

### Command Line Interface

<!-- sample CLI usage -->

## Demo

<!-- describe each provided sample ipynb -->

### Interactive Demo

<!-- link to the interactive run demo -->

## Todo

 - [ ] Handle custom Markdown commands to override the module tree structure.
 - [ ] Fix attribute name definitions conflicting with globally tracked identifiers (refer to `sample_HiveNAS.ipynb` -> `foodsource.py`'s `time` property conflicting with the `time` package, leading to an improper `import time` dependency).
 - [ ] Reimplement the Processor logic to a 2-pass approach to infer the module tree structure and node types, *then* analyze the usages/dependencies (the CDA bit). This is to overcome the current hacky relative-imports' solution for package-level modules (refer to [./fileops/writer.py:__write_module_node()](https://github.com/ThunderStruct/PyNBRefactor/blob/7dfbf751d9b05e99fc5aedf6e3b729bf7299b0c8/fileops/writer.py#L38) for the full description).

## License

PyNBRefactor is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

PRs are welcome (and encouraged)! If you'd like to contribute to PyNBRefactor, please read the [CONTRIBUTING](https://github.com/ThunderStruct/PyNBRefactor/blob/main/CONTRIBUTING.md) guidelines.


