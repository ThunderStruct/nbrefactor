""" Refactors Jupyter Notebooks into Python Packages and vice-versa
"""

from utils.options import get_args
from utils.parser import parse_notebook
from utils.injector import write_modules
from utils.cda import analyze_dependencies


def refactor_notebook(notebook_path, output_path):
    """
    Refactor a Jupyter notebook into a hierarchical package structure.

    Args:
    notebook_path (str): Path to the Jupyter notebook.
    output_path (str): Path to the output directory.
    """
    imports, code_cells = parse_notebook(notebook_path)

    print('\n'.join([key for key in code_cells.keys()]))

    dependencies = analyze_dependencies(code_cells, imports)
    write_modules(dependencies, output_path)


if __name__ == '__main__':
    args = get_args()
    print('args', args)
    refactor_notebook(args.notebook_path, args.output_path)
