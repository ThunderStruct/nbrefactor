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
    
    if not node.children:
        # leaf node -> this is a module, write file
        filename = f'{node.name}.py'
        file_path = os.path.join(output_path, filename)
        with open(file_path, 'w') as f:
            for parsed_cell in node.parsed_cells:
                if isinstance(parsed_cell, ParsedCodeCell):
                    # inject imports / dependencies
                    for dependency in parsed_cell.dependencies:
                        f.write(dependency + '\n')
                    f.write('\n' + parsed_cell.parsed_source)
    else:
        # node has children -> create dir
        module_path = os.path.join(output_path, node.name)
        ensure_dir(module_path)
        
        # node could potentially have code at package-level, so...
        if node.has_code_cells():
            filename = f'{node.name}.py'
            file_path = os.path.join(module_path, filename)
            with open(file_path, 'w') as f:
                for parsed_cell in node.parsed_cells:
                    if isinstance(parsed_cell, ParsedCodeCell):
                        # inject imports / dependencies
                        for dependency in parsed_cell.dependencies:
                            f.write(dependency + '\n')
                        f.write('\n' + parsed_cell.parsed_source)

        # recursively write child nodes
        for child in node.children.values():
            write_modules(child, module_path)
        