
import sys
sys.path.append('..')

from datastructs import ModuleNode
from filesys import read_notebook, write_modules
from .parser import parse_code_cell, parse_markdown_cell

def process_notebook(notebook_path, output_path):

    # read nb
    unparsed_cells = read_notebook(notebook_path)
    
    # build module tree
    root = ModuleNode('root')
    current_node = root
    node_stack = [root]
    
    # track declared definitions
    global_definitions = {}

    # parse all cells
    for cell in unparsed_cells:
        if cell.cell_type == 'markdown':
            parsed_md = parse_markdown_cell(cell.cell_idx, cell.raw_source)
            for header in parsed_md.headers:
                header_name = header.name.replace(' ', '_').lower()

                # Adjust the stack and current_node based on the header level
                while len(node_stack) > header.level:
                    node_stack.pop()

                # Create a new node if it doesn't exist
                if header_name not in node_stack[-1].children:
                    new_node = ModuleNode(header_name, node_stack[-1])
                    node_stack[-1].add_child(new_node)
                    node_stack.append(new_node)
                else:
                    # Move to the existing node
                    node_stack.append(node_stack[-1].children[header_name])

                current_node = node_stack[-1]

            current_node.add_parsed_cell(parsed_md)

        elif cell.cell_type == 'code':
            parsed_code = parse_code_cell(cell.cell_idx, cell.raw_source, global_definitions)
            current_node.add_parsed_cell(parsed_code)

    # write
    write_modules(root, output_path)