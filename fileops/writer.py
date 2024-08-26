""" Writes the analyzed modules / notebook
"""

import sys
sys.path.append('..')

import os
from .utils import ensure_dir
from datastructs import ParsedCodeCell

def write_modules(node, output_path):
    """
    Recursively writes the modules and their contents given a :class:`~ModuleNode` tree hierarchy.
    
    Args:
        node (ModuleNode): the current node representing a module or package.
        output_path (str): the path where the Python files should be written.
    """
    
    if not node.children and node.has_code_cells():
        # leaf node -> this is a module, write file
        __write_module_node(node, output_path, is_package_level=False)

    else:
        # node has children -> create dir
        module_path = os.path.join(output_path, node.name)
        ensure_dir(module_path)

        # node could potentially have code at package-level, so...
        if node.has_code_cells():
            __write_module_node(node, module_path, is_package_level=True)

        # recursively write child nodes
        for child in node.children.values():
            write_modules(child, module_path)
        

def __write_module_node(node, output_path, is_package_level=False):
    """
    TODO: this is a hacky way of fixing package-level relative imports. 
    Need to think of a more elegant solution

    We use Markdown headers as both directories (packages) and file names (modules).
    The autonmous, non-command-based method to distinguish between the two is checking
    if the node in the tree has children (leaf node). If it does -> it is a dir, 
    else -> it is a module

    The problem is that a package-level modules 
    (i.e. , ./root/package_a/package_a.py, where package_a 
    contains sub-packages as well) are treated as directories 
    in the module tree rather than files.

    Example notebook:

    cell 0: ## Utilities
    cell 1: *code*
    cell 2: ### File Utils
    cell 3: *code*
    ...

    This translates into 

    -root
    --utilities
    ---utilties.py (cell 1's code)
    ---file_utils.py (cell 3's code)

    In this example, utilities.py is what I refer to as package-level modules. 
    It is treated as the same depth as `cell 0`, 
    hence it is missing an up-level '.' in the relative import. 
    
    This is more challenging than it sounds because we parse the notebook sequentially, 
    so there is no way to figure out if the package at ## Utilities is a package 
    or a module until we reach cell 2 (### File Utils). By which time, 
    cell 1's code was parsed, analyzed, and tracked. 
    
    I hate to think it but... We might need to restructure the parsing logic to do 2 passes, 
    first one figures out packages/modules, second one parses. 
    
    For now, we bite the bullet and add a '.' manually (I won't be able to sleep peacefully)
    """

    filename = f'{node.name}.py'
    file_path = os.path.join(output_path, filename)

    with open(file_path, 'w') as f:
        
        # inject imports / dependencies
        for dependency in node.aggregate_dependencies():
            if is_package_level and dependency.startswith('from .'):
                hacky_dependency = dependency.replace('from .', 'from ..')
                f.write(hacky_dependency + '\n')
            else:
                f.write(dependency + '\n')
                
        # write source
        for parsed_cell in node.parsed_cells:
            if isinstance(parsed_cell, ParsedCodeCell):
                f.write('\n' + parsed_cell.parsed_source + '\n\n')

