
from collections import defaultdict

def build_module_tree(parsed_cells, root_package='', 
                      default_filename='default.py'):
    """
    Constructs a tree structure from the inferred parsed notebook cells.

    Args:
        parsed_cells (list): a list of ParsedCell objects extracted from \
            the given notebook.
        root_package (str, optional): Root package name, use '' for the \
            current dir.
        default_filename (str, optional): The default filename if none \
            is given at the start of a cell (i.e. `filename: foobar.py`).

    """
    
    imports = set([])
    code_cells = defaultdict(list)
    current_hierarchy = [root_package]
    filename = default_filename


    # imports.update(set(parsed_code.imports))      # extend imports set
    # code_cells[os.sep.join(current_hierarchy) \
    #             + os.sep + filename].append(parsed_code.parsed_source)