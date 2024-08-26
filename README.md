
# PyNBRefactor

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


The core of PyNBRefactor's functionality lies in its sophisticated Code Dependency Analyzer (CDA). The CDA is responsible for parsing code, tracking declared definitions, and analyzing dependencies across the generated modules. It works in the following steps:


## Todo

 - [ ] Handle custom Markdown commands to override the module tree structure.
 - [ ] Fix attribute name definitions conflicting with globally tracked identifiers (refer to `sample_HiveNAS.ipynb` -> `foodsource.py`'s `time` property conflicting with the `time` package, leading to an improper `import time` dependency).
 - [ ] Reimplement the Processor logic to a 2-pass approach to infer the module tree structure and node types, *then* analyze the usages/dependencies (the CDA bit). This is to overcome the current hacky relative-imports' solution for package-level modules (refer to `./fileops/writer.py:__write_module_node()` for the full description).

## License

PyNBRefactor is licensed under the MIT License. See the [LICENSE](https://github.com/ThunderStruct/PyNBRefactor/blob/main/LICENSE) file for more details.

## Contributing

PRs are welcome (and encouraged)! If you'd like to contribute to PyNBRefactor, please read the [CONTRIBUTING](https://github.com/ThunderStruct/PyNBRefactor/blob/main/CONTRIBUTING.md) guidelines.


