""" Writes the analyzed modules / notebook
"""

import sys
sys.path.append('..')

import os
import astor
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
                    for imp in parsed_cell.imports:
                        f.write(astor.to_source(imp))
                    f.write(parsed_cell.source + '\n')
    else:
        # node has children -> create dir
        module_path = os.path.join(output_path, node.name)
        if not os.path.exists(module_path):
            os.makedirs(module_path)
        
        for child in node.children.values():
            write_modules(child, module_path)