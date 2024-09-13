""" File reader
"""

import nbformat
from ..datastructs import UnparsedCell
from .utils import ensure_ipynb_validity

def read_notebook(notebook_path):
    """
    Reads the notebook into UnparsedCell objects, separating the code
    cells from markdown cells.
    
    Args:
        notebook_path (str): relative path to the Jupyter notebook.
    Returns:
        list: a list of UnparsedCell objects containing both the
        code cells and markdown cells.
    Raises:
        FileNotFoundError: If `notebook_path` does not exist.
        ValueError: If `notebook_path` does not point to an .ipynb file.
    """

    path = ensure_ipynb_validity(notebook_path)     # raises error if invalid
    
    with open(path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    unparsed_cells = []
    for cell_idx, cell in enumerate(nb.cells):
        unparsed_cells.append(UnparsedCell(cell_idx, 
                                           cell.cell_type, 
                                           cell.source))

    return unparsed_cells