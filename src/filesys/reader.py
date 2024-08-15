""" Parsing methods for both notebooks and packages/modules
"""

import sys
sys.path.append('..')

import nbformat
from datastructs import UnparsedCell

def read_notebook(notebook_path):
    """
    Reads the notebook into UnparsedCell objects, separating the code
    cells from markdown cells.
    
    Args:
        notebook_path (str): relative path to the Jupyter notebook.
    Returns:
        list: A list of UnparsedCell objects containing both the
        code cells and markdown cells.
    """

    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    unparsed_cells = []
    for cell_idx, cell in enumerate(nb.cells):
        unparsed_cells.append(UnparsedCell(cell_idx, cell.cell_type, cell.source))

    return unparsed_cells