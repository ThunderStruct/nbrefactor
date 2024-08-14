
from collections import defaultdict
from ..filesys import read_notebook, read_modules
from ..parser import parse_notebook_code, parse_notebook_markdown

def refactor_notebook(notebook_path, root_package='', 
                      default_filename='default.py'):
    """
    """

    # Read in the notebook
    unparsed_cells = read_notebook(notebook_path)

    # Parse cells in order + resolve dependencies
    for cell in unparsed_cells:
        if cell.cell_type == 'code':
            # parse and format code cells
            parsed_code = parse_notebook_code(cell.cell_idx, cell.source)
            parsed_cells.append(parsed_code)
        
        elif cell.cell_type == 'markdown':
            # markdown; setting code-cell hierarchical level
            parsed_markdown = parse_notebook_markdown(cell.cell_idx, cell.source)
            parsed_cells.append(parsed_markdown)
        

    # imports = set([])
    # code_cells = defaultdict(list)
    # current_hierarchy = [root_package]
    # filename = default_filename


    # # imports.update(set(parsed_code.imports))      # extend imports set
    # # code_cells[os.sep.join(current_hierarchy) \
    # #             + os.sep + filename].append(parsed_code.parsed_source)




