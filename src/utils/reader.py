""" Parsing methods for both notebooks and packages/modules
"""

import os
import nbformat
from collections import defaultdict
from .parser import parse_notebook_code, parse_notebook_markdown

def read_notebook(notebook_path, default_filename='default.py',
                  root_package=''):
    '''
    Reads the notebook into ParsedCell objects, separating the code \
    cells from markdown cells and parsing the MD header levels/command \
    directives.

    Args:
        notebook_path (str): Path to the Jupyter notebook.
    Returns:
        list: A list of ParsedCell objects containing both the parsed \
        code cells and parsed markdown cells.
    '''
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)


    parsed_cells = []
    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # parse and format code cells
            parsed_code = parse_notebook_code(cell_idx, cell.source)
            parsed_cells.append(parsed_code)
        
        elif cell.cell_type == 'markdown':
            # markdown; setting code-cell hierarchical level
            parsed_markdown = parse_notebook_markdown(cell_idx, cell.source)
            parsed_cells.append(parsed_markdown)

    return parsed_cells


def read_modules(source_dir):
    """
    Crawl hierarchically through a given source directory
    parsing and consolidating 
    """

    pass