""" File-writing methods for a parsed/refactored notebook
"""

import os
from .utils import ensure_dir
from ..datastructs import ParsedCodeCell


def write_modules(node, output_path, pre_write_hook=None, generate_init=False):
    """
    Recursively writes the modules and their contents given a \
        :class:`~nbrefactor.datastructs.ModuleNode` tree hierarchy.
    
    Args:
        node (ModuleNode): the current node representing a module or package.
        output_path (str): the path where the Python files should be written.
        write_hook (callable, optional): a hook function that takes \
            (content, node) and returns the modified content.
        generate_init (bool, optional): whether to generate __init__.py files \
            in package directories. Defaults to False.
    """
    
    if (not node.children and node.has_code_cells()) \
        or node.node_type == 'module':
        # leaf node -> this is a module, write file OR
        # asserted 'module' type -> write file
        __write_module_node(node, output_path, 
                            is_package_level=False, 
                            pre_write_hook=pre_write_hook)

    else:
        if node.ignore_package:
            # prune this branch
            return
        
        # node has children -> create dir
        module_path = os.path.join(output_path, node.name)
        ensure_dir(module_path)

        # create `__init__.py` file for package if requested
        if generate_init and node.name != '.':  # skip root directory
            init_path = os.path.join(module_path, '__init__.py')
            with open(init_path, 'w') as f:
                # write package docstring
                f.write('"""Package initialization."""\n\n')
                
                # if this node has code cells, expose them at package level
                if node.has_code_cells():
                    module_name = os.path.splitext(node.name)[0]
                    # noqa to supress import warnings (refer to PEP 8)
                    f.write(f'from .{module_name} import *  # noqa\n')
                
                # expose all child modules
                for child in node.children.values():
                    if not child.ignore_package and not child.ignore_module:
                        child_name = os.path.splitext(child.name)[0]
                        # noqa to supress import warnings
                        f.write(f'from .{child_name} import *  # noqa\n')

        # node could potentially have code at package-level, so...
        if node.has_code_cells():
            __write_module_node(node, module_path, 
                                is_package_level=True,
                                pre_write_hook=pre_write_hook)

        # recursively write child nodes
        for child in node.children.values():
            write_modules(child, module_path, 
                         pre_write_hook=pre_write_hook,
                         generate_init=generate_init)
        

def __write_module_node(node, output_path, 
                        is_package_level=False, pre_write_hook=None):
    """
    TODO: this is a hacky way of fixing package-level relative imports. 
    Need to think of a more elegant solution

    We use Markdown headers as both directories (packages) and \
        file names (modules).
    The autonmous, non-command-based method to distinguish between the two \
        is checking
    if the node in the tree has children (leaf node). \
        If it does -> it is a dir, 
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
    
    This is more challenging than it sounds because we parse the notebook \
        sequentially, 
    so there is no way to figure out if the package at ## Utilities is a \
    package 
    or a module until we reach cell 2 (### File Utils). By which time, 
    cell 1's code was parsed, analyzed, and tracked. 
    
    I hate to think it but... We might need to restructure the parsing logic \
        to do 2 passes, 
    first one figures out packages/modules, second one parses. 
    
    For now, we bite the bullet and add a '.' manually (I won't be able to \
        sleep peacefully)


    Args:
        node (ModuleNode): the current node representing a module or package.
        output_path (str): the path where the Python files should be written.
        is_package_level (bool, optional): whether this node is a \
            package-levle module (i.e. node has sub-packages + code cell(s). \
            This creates a dir AND a `.py` module).
        write_hook (callable, optional): a hook function that takes \
            (content, node) and returns the modified content.
    """
    
    if node.ignore_module:
        return
    
    filename = f'{os.path.splitext(node.name)[0]}.py'   # assert .py extension 
                                                        # if it does not exist
    file_path = os.path.join(output_path, filename)

    with open(file_path, 'w') as f:
        content = []

        # inject imports / dependencies
        for dependency in node.aggregate_dependencies():
            if is_package_level and dependency.startswith('from .'):
                hacky_dependency = dependency.replace('from .', 'from ..')
                content.append(hacky_dependency)
            else:
                content.append(dependency)
                
        # write source
        for parsed_cell in node.parsed_cells:
            if isinstance(parsed_cell, ParsedCodeCell):
                content.append(parsed_cell.parsed_source + '\n\n')
        
        file_content = '\n'.join(content)

        # apply pre-writing hook if applicable
        if pre_write_hook:
            file_content = pre_write_hook(file_content, node)

        f.write(file_content)


        

