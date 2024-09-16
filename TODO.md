
# Todo

- 🔴 High priority _(i.e. critical and urgent features / fixes)_
- 🟠 Medium priority _(i.e. important, but not detrimental to normal operation)_
- 🟢 Low priority _(i.e. "would be nice to have")_


## Major

- [ ] 🔴 Fix attribute name definitions conflicting with globally tracked identifiers (refer to [sample_HiveNAS.ipynb](src/demo/examples/sample_HiveNAS.ipynb) -> `foodsource.py`'s `time` property conflicting with the `time` package, leading to an improper `import time` dependency).

- [ ] 🔴 Add a feature to use the CDA to exclude non-functional modules (i.e. all comments or `pass` statements. Those could be resulting from a notebook cell that exclusively contain magic commands (which get replaced by `pass # \1`), yielding an essentially empty file).

- [ ] 🔴 Update the CDA to handle global (cell-level) variable tracking and injection (we currently only track functions' and classes' definitions). The most elegant approach I could think of at the moment that does not involve `globals()` is to create a root-level module `global_vars.py` or so and accumulate all definitions there.

- [ ] 🟠 Reimplement the Processor logic to a 2-pass approach to infer the module tree structure and node types, _then_ analyze the usages/dependencies (the CDA bit). This is to overcome the current hacky relative-imports' solution for package-level modules (refer to [./fileops/writer.py:\_\_write_module_node()](https://github.com/ThunderStruct/nbrefactor/blob/63322fe4a33422d2982dedbd3683ee1e2f9bc739/src/nbrefactor/fileops/writer.py#L49) for the full description).

- [ ] 🟢 Add and handle custom Markdown commands to override the module tree structure (e.g. assign node IDs, force parent nodes regardless of Markdown structure, node depth (partially implemented in the [processor](https://github.com/ThunderStruct/nbrefactor/blob/b2eb3b501da898c625cb1712c076df8b0ac3896e/src/nbrefactor/processor/processor.py#L287)), etc.).


## Minor

- [ ] 🔴 Add a CLI flag to instantiate modules (add `__init__.py` to sub-directories in the writer phase).


## Patches

- [ ] 🟠 Add a "before and after" example to the docs

- [ ] 🟢 Restructure the docs' `index.rst` to have a more informative homepage than just the toctree. Also add the version number (already declared in [conf.py](docs/source/conf.py) and accessible through `|version|` in RST).

