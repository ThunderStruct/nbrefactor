""" Refactors Jupyter Notebooks into Python Packages
"""

from utils.options import Options
from processor import process_notebook


def refactor_notebook(notebook_path, output_path):
    """
    Refactor a Jupyter notebook into a hierarchical package structure.

    Args:
    notebook_path (str): Path to the Jupyter notebook.
    output_path (str): Path to the output directory.
    """

    root_node = process_notebook('./src/notebook.ipynb', './refactored/')


if __name__ == '__main__':
    Options.parse_args()
    print('args', Options.get_options())
    refactor_notebook(Options.get('notebook_path'), 
                      Options.get('output_path'))
