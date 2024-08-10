""" Parsing methods for both notebooks and packages/modules
"""

import os
import nbformat
from collections import defaultdict
from .parser import parse_notebook_code, parse_notebook_markdown

def read_notebook(notebook_path, default_filename='default.py',
                  root_package='', auto_module_hierarchy=False):
    '''
    Parse the notebook to extract import statements and code cells.

    Args:
    notebook_path (str): Path to the Jupyter notebook.
    default_filename (str, optional): The default filename if none \
        is given at the start of a cell (i.e. `filename: foobar.py`).
    root_package (str, optional): Root package name, '' for the \
        current dir.
    auto_module_hierarchy (bool, optional): whether or not to infer \
        module hierarchy from the Markdown header level. Defaults to \
        `False`, and Markdown directives will override this even if \
        set to `True`.

    Returns:
    tuple: A tuple containing a parsed list of import lines and a parsed \
        dict of code cells (dot-separated keys representing the package \
        hierarchy).
    '''
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    imports = set([])
    code_cells = defaultdict(list)
    current_hierarchy = [root_package]
    filename = default_filename

    for cell in nb.cells:
        if cell.cell_type == 'code':
            parsed_imports, parsed_code = parse_notebook_code(cell.source)

            imports.update(set(parsed_imports))      # extend imports set
            code_cells[os.sep.join(current_hierarchy) + os.sep + filename].append(parsed_code)
        
        elif cell.cell_type == 'markdown':
            # markdown; setting code-cell hierarchical level
            parse_notebook_markdown(cell.source, current_hierarchy)

    return imports, code_cells


def read_modules(source_dir):
    """
    Crawl hierarchically through a given source directory
    parsing and consolidating 
    """

    pass