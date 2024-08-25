"""

"""

import re
import sys
sys.path.append('..')

from datastructs import ModuleNode, ParsedCodeCell
from datastructs import MarkdownHeader, MarkdownCommand, MarkdownCommandType
from fileops import read_notebook, write_modules
from .parser import parse_code_cell, parse_markdown_cell
 

def process_notebook(notebook_path, output_path, root_package='.'):
    # read notebook
    unparsed_cells = read_notebook(notebook_path)
    
    # init the module tree
    root = ModuleNode(root_package)
    current_node = root
    node_stack = [root]

    # parse all cells
    for cell in unparsed_cells:
        
        if cell.cell_type == 'markdown':
            # MARKDOWN CELL
            parsed_md = parse_markdown_cell(cell.cell_idx, cell.raw_source)
            for md_element in parsed_md.elements:
                if isinstance(md_element, MarkdownHeader):
                    # handle MarkdownHeader
                    header = md_element
                    header_name = __sanitize_node_name(header.name)

                    while len(node_stack) > header.level + 1:
                        node_stack.pop()

                    # if we're on the same level as the stack's current node,
                    # we move back up to the correct parent level (without popping the root node)
                    if len(node_stack) == header.level + 1 and len(node_stack) > 1:
                        node_stack.pop()

                    new_node = ModuleNode(header_name, node_stack[-1])
                    node_stack[-1].add_child(new_node)
                    node_stack.append(new_node)

                    current_node = new_node

                elif isinstance(md_element, MarkdownCommand):
                    # handle MarkdownCommand
                    __handle_markdown_command(md_element, current_node, node_stack)
                    
            current_node.add_parsed_cell(parsed_md)

        elif cell.cell_type == 'code':
            # CODE CELL
            parsed_code = parse_code_cell(cell.cell_idx, cell.raw_source, current_node)
            current_node.add_parsed_cell(parsed_code)
    
    # write the parsed module tree
    write_modules(root, output_path)

    return root


def __handle_markdown_command(command, current_node, node_stack):
    """
    Processes a MarkdownCommand and adjusts the module tree accordingly.

    This will override the previous header if present.

    Example Markdown:

        # Root
        ## Some Header
        <!--- 
            $module=my_module.py 
        -->

        ## Package
        <!--- $package=my_package -->

        ### subpackage
        #### Sub-module

    This will resolve into:
        - root
        --my_module.py
        --my_package
        ---subpacakge
        ----sub_module.py

    Args:
        command (MarkdownCommand): the command to process
        current_node (ModuleNode): the current module node in the tree
        node_stack (list): the stack representing the current path in the module tree
    """
    if command.type == MarkdownCommandType.DECLARE_MODULE:
        # override the current module's name with the specified module name
        new_node_name = command.value.replace('.', '').replace(' ', '_').replace('-', '_').lower()
        new_node = ModuleNode(new_node_name, current_node.parent)
        new_node.set_node_type('module')
        current_node.parent.add_child(new_node)
        node_stack[-1] = new_node

    elif command.type == MarkdownCommandType.DECLARE_PACAKGE:
        # create a new package node at the current level
        package_name = command.value.replace('.', '').replace(' ', '_').replace('-', '_').lower()
        new_node = ModuleNode(package_name, current_node)
        new_node.set_node_type('package')
        current_node.add_child(new_node)
        node_stack.append(new_node)
        current_node = new_node

    elif command.type == MarkdownCommandType.NODE_DEPTH:
        # adjust the node depth (if applicable)
        desired_depth = int(command.value)
        while len(node_stack) > desired_depth + 1:
            node_stack.pop()
        current_node = node_stack[-1]

    elif command.type == MarkdownCommandType.IGNORE_MODULE:
        # pop the stack to ignore a single module
        node_stack.pop()
        current_node = node_stack[-1]

    elif command.type == MarkdownCommandType.IGNORE_PACKAGE:
        # ignore the current package (all ModuleNodes under it)
        while len(node_stack) > 1:
            node_stack.pop()
        current_node = node_stack[-1]

    elif command.type == MarkdownCommandType.IGNORE_CELL:
        # this is only here for completeness
        pass


def __sanitize_node_name(node_name, default_name='unnamed'):
    """
    Sanitizes a given node name (typically a Markdown header name).

    Args:
        node_name (str): the given (potentially invalid) node name

    Returns:
        str: sanitized/valid filename
    """

    node_name = node_name.replace(' ', '_').replace('-', '_').lower()
    node_name = re.sub(r'[^a-z0-9_]', '', node_name)

    # trim leading/trailing underscores
    node_name = node_name.strip('_')

    return node_name or default_name