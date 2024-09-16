
# Todo


## Major

- [ ] $${\color{red}critical}$$ Fix attribute name definitions conflicting with globally tracked identifiers (refer to `sample_HiveNAS.ipynb` -> `foodsource.py`'s `time` property conflicting with the `time` package, leading to an improper `import time` dependency).

- [ ] Handle custom Markdown commands to override the module tree structure.


## Minor

- [ ] Add a CLI flag to instantiate modules (add `__init__.py` to sub-directories in the writer phase).

## Patches



- [ ] Reimplement the Processor logic to a 2-pass approach to infer the module tree structure and node types, _then_ analyze the usages/dependencies (the CDA bit). This is to overcome the current hacky relative-imports' solution for package-level modules (refer to [./fileops/writer.py:\_\_write_module_node()](https://github.com/ThunderStruct/nbrefactor/blob/7dfbf751d9b05e99fc5aedf6e3b729bf7299b0c8/fileops/writer.py#L38) for the full description).
- [ ] Add a feature to use the CDA to exclude non-functional modules (i.e. all comments or `pass` statements. Those could be resulting from a notebook cell that exclusively contain magic commands (which get replaced by `pass # \1`), yielding an essentially empty file).
- [ ] Update the CDA to handle global (cell-level) variable tracking and injection (we currently only track functions' and classes' definitions). The most elegant approach I could think of at the moment that does not involve `globals()` is to create a root-level module `global_vars.py` or so and accumulate all definitions there.