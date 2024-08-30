"""

"""

import re
import sys
from time import sleep
sys.path.append('..')

from tqdm.auto import tqdm

from datastructs import ModuleNode
from datastructs import MarkdownHeader, MarkdownCommand, MarkdownCommandType
from fileops import read_notebook, write_modules
from .parser import parse_code_cell, parse_markdown_cell
 

def process_notebook(notebook_path, output_path, root_package='.'):
    
    # read notebook
    print(f'Reading notebook at ({notebook_path})...')
    sleep(0.25)     # for aesthetic purposes 0:)
    unparsed_cells = read_notebook(notebook_path)       # exits if file does not exist/is invalid
    print(f'Reading complete!\n')
    
    # init the module tree
    root = ModuleNode(root_package)
    current_node = root
    node_stack = [root]
    accumulated_warnings = []

    # parse all cells
    for cell in tqdm(unparsed_cells, desc='Processing Notebook'):
        
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

                elif isinstance(md_element, MarkdownCommand):
                    # handle MarkdownCommand
                    __handle_markdown_command(md_element, current_node, node_stack)
                
                current_node = node_stack[-1]       # update current node 
                                                    # (potentially manipulated through MD headers or commands)
                current_node.add_parsed_cell(parsed_md)
                accumulated_warnings.extend(parsed_md.warnings)

        elif cell.cell_type == 'code':
            # CODE CELL
            if current_node.ignore_package or current_node.ignore_module or current_node.ignore_next_cell:
                # avoid parsing (unnecessary cost + we don't want any definitions in 
                # this cell tracked in the CDA)
                current_node.ignore_next_cell = False
                continue

            parsed_code = parse_code_cell(cell.cell_idx, cell.raw_source, current_node)
            current_node.add_parsed_cell(parsed_code)
        sleep(0.025)         # for aesthetic purposes 0:)
    
    # update the tree to prune out ignored branches
    print('Flushing...\n')
    sleep(0.25)     # for aesthetic purposes 0:)
    __flush_pruned_nodes(root)

    # write the parsed module tree
    print('Writing modules...')
    sleep(0.25)     # for aesthetic purposes 0:)
    write_modules(root, output_path)
    print('Writing complete!')

    # log warnings
    print('\n--------------------')
    print(f'Warnings: ({len(accumulated_warnings)})\n')
    for warning in accumulated_warnings:
        print(f'\t{warning["source"]} (Cell #{warning["cell_idx"]}): {warning["message"]}\n')

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

        new_node_parent = current_node

        # default to sibling-level if we're not at root level
        if current_node.parent is not None:    
            node_stack.pop()
            new_node_parent = current_node.parent

        new_node = ModuleNode(node_name, new_node_parent, 
                              depth=new_node_parent.depth + 1)
        
        new_node.node_type = 'module'
        new_node_parent.add_child(new_node)

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


def __flush_pruned_nodes(node):
    """
    Recursively traverse the tree from the root to flush out 
    ignored/pruned packages.

    Args:
        node (ModuleNode): the current node representing a module or package.
    """

    node.children = dict((k, v) for (k, v) in node.children.items() if not v.ignore_package and not v.ignore_module)

    for _, child in node.children.items():
        __flush_pruned_nodes(child)

