""" Refactors Jupyter Notebooks into Python Packages and vice-versa
"""

from utils.options import Options
from utils.reader import read_notebook
from utils.injector import write_modules


def refactor_notebook(notebook_path, output_path):
    """
    Refactor a Jupyter notebook into a hierarchical package structure.

    Args:
    notebook_path (str): Path to the Jupyter notebook.
    output_path (str): Path to the output directory.
    """
    imports, code_cells = read_notebook(notebook_path)

    print('\n'.join([key for key in code_cells.keys()]))

    dependencies = analyze_dependencies(code_cells, imports)
    write_modules(dependencies, output_path)


if __name__ == '__main__':
    Options.parse_args()
    print('args', Options.get_options())
    refactor_notebook(Options.get('notebook_path'), 
                      Options.get('output_path'))
