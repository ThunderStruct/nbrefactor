""" Writes the analyzed modules / notebook
"""

import sys
sys.path.append('..')

import os
import astor
from datastructs import ParsedCodeCell

def write_modules(node, output_path):
    """
    Recursively writes the modules and their contents to disk.
    
    Args:
        node (ModuleNode): The current node representing a module or package.
        output_path (str): The path where the Python files should be written.
    """
    module_path = os.path.join(output_path, *node.get_full_path())

    if not os.path.exists(module_path):
        os.makedirs(module_path)

    # write all parsed cells to respective module node
    for parsed_cell in node.parsed_cells:
        if isinstance(parsed_cell, ParsedCodeCell):
            with open(os.path.join(module_path, f'cell_{parsed_cell.cell_idx}.py'), 'w') as f:
                # inject imports + definitions + source
                for imp in parsed_cell.imports:
                    f.write(astor.to_source(imp) + '\n')
                f.write(parsed_cell.source + '\n')

    # recursively module nodes
    for child in node.children.values():
        write_modules(child, output_path)