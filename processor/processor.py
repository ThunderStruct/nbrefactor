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

            if any([True for e in parsed_md.elements \
                    if isinstance(e, MarkdownCommand) \
                        and e.type == MarkdownCommandType.IGNORE_MARKDOWN and e.value]):
                # ignore this entire Markdown cell if an $ignore-markdown command is present
                continue

            for md_element in parsed_md.elements:
                if isinstance(md_element, MarkdownHeader):
                    # handle MarkdownHeader
                    header = md_element
                    header_name = __sanitize_node_name(header.name)

                    new_depth = header.level
                    current_depth = current_node.depth

                    # infer node position / depth
                    if new_depth > current_depth:
                        # need to move deeper -> add child node
                        new_node = ModuleNode(header_name, current_node, depth=new_depth)
                        current_node.add_child(new_node)
                        node_stack.append(new_node)
                    elif new_depth == current_depth:
                        # same level -> replace current node
                        node_stack.pop()
                        new_node = ModuleNode(header_name, current_node.parent, depth=new_depth)
                        current_node.parent.add_child(new_node)
                        node_stack.append(new_node)
                    else:
                        # need to move up th hierarchy -> pop the stack until target depth is reached
                        while node_stack and node_stack[-1].depth >= new_depth:
                            node_stack.pop()
                        new_node = ModuleNode(header_name, node_stack[-1], depth=new_depth)
                        node_stack[-1].add_child(new_node)
                        node_stack.append(new_node)

                    current_node = new_node

                elif isinstance(md_element, MarkdownCommand):
                    # handle MarkdownCommand
                    __handle_markdown_command(md_element, current_node, node_stack)
                    
            current_node.add_parsed_cell(parsed_md)

        elif cell.cell_type == 'code':
            # CODE CELL
            if current_node.ignore_package or current_node.ignore_module or current_node.ignore_next_cell:
                # avoid parsing (unnecessary cost + we don't want any definitions in 
                # this cell tracked in the CDA)
                current_node.ignore_next_cell = False
                continue

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
            $node=my_module
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

    # NODE-SKIPPING COMMANDS
    if command.type == MarkdownCommandType.IGNORE_PACKAGE:
        # ignore the current node (and all upcoming nodes beneath it)
        # until a (<= depth) header is encountered
        current_node.ignore_package = True

    elif command.type == MarkdownCommandType.IGNORE_MODULE:
        # ignore all upcoming code cells associated with the current module
        current_node.ignore_module = True

    elif command.type == MarkdownCommandType.IGNORE_CELL:
        # ignore the next code-cell, regardless of node type
        current_node.ignore_next_cell = True

    elif command.type == MarkdownCommandType.IGNORE_MARKDOWN:
        # ignore the current Markdown cell (e.g. instruction cell 
        # that should not be factored into the hierarchy)

        # this is handled externally to avoid the processing cost
        pass

    # NODE-MANIPULATION COMMANDS
    elif command.type == MarkdownCommandType.RENAME_PACKAGE:
        # override the current node's name + assert package type
        current_node.name = __sanitize_node_name(command.value)
        current_node.node_type = 'package'

    elif command.type == MarkdownCommandType.RENAME_MODULE:
        # override the current node's name + assert module type
        current_node.name = __sanitize_node_name(command.value)
        current_node.node_type = 'module'

    elif command.type == MarkdownCommandType.RENAME_NODE:
        # override the current node's name 
        current_node.name = __sanitize_node_name(command.value)

    # NODE-DECLARATION COMMANDS
    elif command.type == MarkdownCommandType.DECLARE_PACKAGE:
        # create a new node and assert its node type to package
        node_name = __sanitize_node_name(command.value)
        new_node = ModuleNode(node_name, current_node, 
                              depth=current_node.depth + 1)    # child-level
        new_node.node_type = 'package'
        current_node.add_child(new_node)
        node_stack.append(new_node)

    elif command.type == MarkdownCommandType.DECLARE_MODULE:
        # create a new node and assert its node type to module
        node_name = __sanitize_node_name(command.value)
        is_sibling = current_node.parent is not None    # default to sibling-level if not root level
            
        new_node = ModuleNode(node_name, current_node, 
                              depth=current_node.depth + \
                                (1 if is_sibling else 0))
        new_node.node_type = 'module'
        if is_sibling:
            current_node.parent.add_child(new_node)
        else:
            current_node.add_child(new_node)
        node_stack.append(new_node)

    elif command.type == MarkdownCommandType.DECLARE_NODE:
        # create a new generic node (type will be automatically inferred)
        node_name = __sanitize_node_name(command.value)
        new_node = ModuleNode(node_name, current_node, depth=current_node.depth + 1)
        current_node.add_child(new_node)
        node_stack.append(new_node)

    # POTENTIAL FUTURE IMPLEMENTATIONS
    # TODO: possibly add this in future iterations of the lib, needs more testing for robustness
    # elif command.type == MarkdownCommandType.NODE_DEPTH:
    #     # adjust the node depth (if applicable)
    #     desired_depth = int(command.value)
    #     while len(node_stack) > desired_depth + 1:
    #         node_stack.pop()
    #     current_node = node_stack[-1]



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