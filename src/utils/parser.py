""" Parsing methods for both notebooks and packages/modules
"""

import nbformat
from collections import defaultdict

def __parse_notebook_code_line(line):
    if line.startswith('!'):
        # skip iPython magic commands
        return ''
    
    return line


def parse_notebook(notebook_path, default_filename='default.py',
                   root_package='', 
                   skip_level=['Imports', 'Libraries']):
    '''
    Parse the notebook to extract import statements and code cells.

    Args:
    notebook_path (str): Path to the Jupyter notebook.
    default_filename (str, optional): The default filename if none \
        is given at the start of a cell (i.e. `filename: foobar.py`).
    root_package (str, optional): Root package name, '' for the \
        current dir.
    skip_level (list, optional): a list of header names to skip from \
        indexing. This is primarily used to prevent the "import" or \
        "instructions" headers from being parsed as packages.

    Returns:
    tuple: A tuple containing a parsed list of import lines and a parsed \
        dict of code cells (dot-separated keys representing the package \
        hierarchy).
    '''
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    imports = []
    code_cells = defaultdict(list)
    current_hierarchy = [root_package]
    filename = default_filename

    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            if source.strip().startswith('# filename:'):
                filename = source.split(':')[1].strip()
                continue

            is_ml_comment = False
            code_content = []           # per file

            for line in source.split('\n'):
                # parse imports (whilst handling keywords 'from'/'import' in multiline comments)
                stripped_line = line #.strip()
                if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                    is_ml_comment = not is_ml_comment
                if not is_ml_comment:
                    if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                        imports.append(line)
                    else:
                        code_content.append(__parse_notebook_code_line(line))
                else:
                    code_content.append(__parse_notebook_code_line(line))
            code_cells['>'.join(current_hierarchy) + '>' + filename].append('\n'.join(code_content))
        
        elif cell.cell_type == 'markdown':
            # markdown; setting code-cell hierarchical level
            for line in cell.source.split('\n'):
                if line.startswith('#'):
                    level = line.count('#')
                    header = line.strip('#').strip().replace(' ', '_').lower()
                    
                    if header.lower() in [l.lower() for l in skip_level]:
                        continue

                    if level <= len(current_hierarchy):
                        current_hierarchy = current_hierarchy[:level-1]
                    current_hierarchy.append(header)

    return imports, code_cells


def parse_modules(source_dir):
    """
    Crawl hierarchically through a given source directory
    parsing and consolidating 
    """

    pass