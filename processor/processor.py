"""

"""

import sys
sys.path.append('..')

from datastructs import ModuleNode, MarkdownHeader, MarkdownCommand
from fileops import read_notebook, write_modules
from .parser import parse_code_cell, parse_markdown_cell

def process_notebook(notebook_path, output_path, root_package='.'):

    # read notebook
    unparsed_cells = read_notebook(notebook_path)
    
    # init the module tree
    root = ModuleNode(root_package)
    current_node = root
    node_stack = [root]
    uncommited_nodes = []

    # parse all cells
    for cell in unparsed_cells:
        if cell.cell_type == 'markdown':
            parsed_md = parse_markdown_cell(cell.cell_idx, cell.raw_source)
            for md_element in parsed_md.elements:
                if isinstance(md_element, MarkdownHeader):
                    # handle MarkdownHeader
                    header = md_element
                    header_name = header.name.replace('.', '').replace(' ', '_').replace('-', '_').lower()

                    while len(node_stack) > header.level:
                        node_stack.pop()

                    # if we're on the same level as the stack's current node,
                    # we move back up to the correct parent level
                    if len(node_stack) == header.level:
                        node_stack.pop()

                    new_node = ModuleNode(header_name, node_stack[-1])
                    node_stack[-1].add_child(new_node)
                    node_stack.append(new_node)

                    current_node = new_node

                elif isinstance(md_element, MarkdownCommand):
                    # handle MarkdownCommand

                current_node.add_parsed_cell(parsed_md)

        elif cell.cell_type == 'code':
            parsed_code = parse_code_cell(cell.cell_idx, cell.raw_source, current_node)
            current_node.add_parsed_cell(parsed_code)
    
    # write the parsed module tree
    write_modules(root, output_path)

    return root