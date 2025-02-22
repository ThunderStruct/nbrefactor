""" Notebook-refactoring processor
"""

import re

from tqdm.auto import tqdm

from ..utils import Logger
from ..datastructs import ModuleNode
from ..fileops import read_notebook, write_modules
from .parser import parse_code_cell, parse_markdown_cell
from ..datastructs import MarkdownHeader, MarkdownCommand, MarkdownCommandType


def process_notebook(notebook_path, output_path, 
                     root_package='.', pre_write_hook=None, generate_init=False):
    """
    The Notebook-refactoring entry point. This function:

        1. Reads the notebook into \
            :class:`~nbrefactor.datastructs.UnparsedCell` objects
        2. Parses and processes the unparsed cells into \
            :class:`~nbrefactor.datastructs.ParsedMarkdownCell` and \
                :class:`~nbrefactor.datastructs.ParsedCodeCell` objects \
                    accordingly. \
            This yields a tree of \
                :class:`~nbrefactor.datastructs.ModuleNode` objects \
                representing the resulting file-structure.
        3. Writes the packages and modules given the parsed \
            :class:`~nbrefactor.datastructs.ModuleNode` tree.
    
    Args:
        notebook_path (str): the full path to the Notebook file.
        output_path (str): the desired output path for the refactored project.
        root_package (str, optional): the root package for the refactored \
            project (defaults to `'.'`; i.e. root level of the given \
                `output_path`).
        write_hook (callable, optional): a hook function that takes (content, \
            node) and returns the modified content.
        generate_init (bool, optional): whether to generate __init__.py files \
            in package directories. Defaults to False.
    
    Returns:
        :class:`~nbrefactor.datastructs.ModuleNode`: the root node of the \
            generated tree structure.
        
    """

    # read notebook
    Logger.horizontal_separator(color=Logger.Color.GREEN)
    Logger.log(f'Loading notebook at ({notebook_path})...',
               tag='READING NOTEBOOK', color=Logger.Color.BLUE)
    unparsed_cells = read_notebook(notebook_path)     # raises error if invalid
    Logger.log(f'Loading complete!\n', tag='SUCCESS', 
               color=Logger.Color.GREEN)
    
    # init the module tree
    root = ModuleNode(root_package)
    current_node = root
    node_stack = [root]
    accumulated_warnings = []

    # parse all cells
    for cell in tqdm(unparsed_cells, 
                     desc=(
                         f'{Logger.Color.BLUE}PROCESSING NOTEBOOK'
                         f'{Logger.Color.RESET}')):
        
        if cell.cell_type == 'markdown':
            # MARKDOWN CELL
            parsed_md = parse_markdown_cell(cell.cell_idx, cell.raw_source)

            if any([True for e in parsed_md.elements \
                    if isinstance(e, MarkdownCommand) \
                        and e.type == MarkdownCommandType.IGNORE_MARKDOWN \
                            and e.value]):
                # ignore this entire Markdown cell if an $ignore-markdown 
                # command is present
                continue

            # Check for analyze-only command before processing headers
            analyze_only = any([True for e in parsed_md.elements \
                    if isinstance(e, MarkdownCommand) \
                        and e.type == MarkdownCommandType.ANALYZE_ONLY])

            for md_element in parsed_md.elements:
                if isinstance(md_element, MarkdownHeader):
                    # Skip header processing; we're only analyzing the cell
                    # i.e. do not create a file/folder for it
                    if analyze_only:
                        continue

                    # handle MarkdownHeader
                    header = md_element
                    header_name = __sanitize_node_name(header.name)

                    new_depth = header.level
                    current_depth = current_node.depth

                    # infer node position / depth
                    if new_depth > current_depth:
                        # need to move deeper -> add child node
                        new_node = ModuleNode(header_name, 
                                              current_node, depth=new_depth)
                        current_node.add_child(new_node)
                        node_stack.append(new_node)
                    elif new_depth == current_depth:
                        # same level -> replace current node
                        node_stack.pop()
                        new_node = ModuleNode(header_name, 
                                              current_node.parent, 
                                              depth=new_depth)
                        current_node.parent.add_child(new_node)
                        node_stack.append(new_node)
                    else:
                        # need to move up th hierarchy -> pop the stack until 
                        # target depth is reached
                        while node_stack and node_stack[-1].depth >= new_depth:
                            node_stack.pop()
                        new_node = ModuleNode(header_name, node_stack[-1], 
                                              depth=new_depth)
                        node_stack[-1].add_child(new_node)
                        node_stack.append(new_node)

                elif isinstance(md_element, MarkdownCommand):
                    # handle MarkdownCommand
                    __handle_markdown_command(md_element, current_node, 
                                              node_stack)
                
                current_node = node_stack[-1]       # update current node 
                                                    # (potentially manipulated 
                                                    # through MD headers or commands)
                current_node.add_parsed_cell(parsed_md)
                accumulated_warnings.extend(parsed_md.warnings)

        elif cell.cell_type == 'code':
            # CODE CELL
            if current_node.ignore_package \
                or current_node.ignore_module or current_node.ignore_next_cell:
                # avoid parsing (unnecessary cost + we don't want any 
                # definitions in this cell tracked in the CDA)
                current_node.ignore_next_cell = False
                continue

            parsed_code = parse_code_cell(cell.cell_idx, cell.raw_source, current_node)
            current_node.add_parsed_cell(parsed_code)
    Logger.log(f'Processing complete!\n', tag='SUCCESS', 
               color=Logger.Color.GREEN)

    # update the tree to prune out ignored branches
    Logger.log('Flushing pruned nodes...\n', tag='CLEANING UP', 
               color=Logger.Color.BLUE)
    __flush_pruned_nodes(root)

    # write the parsed module tree
    Logger.log('Writing refactored modules...', tag='FINALIZING', 
               color=Logger.Color.BLUE)

    file_counter = 0    # for logging purposes
    def write_hook_wrapper(content, node):
        nonlocal file_counter
        file_counter += 1

        if pre_write_hook:
            return pre_write_hook(content, node)
        
        return content

    write_modules(root, output_path, 
                  pre_write_hook=write_hook_wrapper,
                  generate_init=generate_init)
    
    Logger.log((
            f'Successfully wrote ({file_counter}) files to "{output_path}"!\n'
        ), 
        tag='SUCCESS', 
        color=Logger.Color.GREEN)

    # log warnings
    Logger.log(f'({len(accumulated_warnings)})', 
               tag='WARNINGS',
               color=Logger.Color.YELLOW)
    
    for warning in accumulated_warnings:
        Logger.log((f'\n\t{warning["source"]} (Cell #{warning["cell_idx"]}): '
              f'{warning["message"]}'))

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
        node_stack (list): the stack representing the current path in the \
            module tree
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

    elif command.type == MarkdownCommandType.ANALYZE_ONLY:
        # Mark the current node to be analyzed but not written to a file
        current_node.analyze_only = True

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

        # If we're in a package context, the module should be created inside that package
        if current_node.node_type == 'package':
            # Create module inside the current package
            new_node = ModuleNode(node_name, current_node, 
                              depth=current_node.depth + 1)
            new_node.node_type = 'module'
            current_node.add_child(new_node)
            node_stack.append(new_node)
        else:
            # Check if this should be a package structure (contains . or /)
            if '.' in node_name or '/' in node_name:
                # Split into package/module parts
                parts = node_name.replace('/', '.').split('.')
                module_name = parts[-1]
                package_parts = parts[:-1]
                
                # Start from current node's parent if we're not at root
                new_node_parent = current_node
                if current_node.parent is not None:
                    node_stack.pop()
                    new_node_parent = current_node.parent
                
                # Create package structure
                for package_part in package_parts:
                    package_node = ModuleNode(package_part, new_node_parent,
                                          depth=new_node_parent.depth + 1)
                    package_node.node_type = 'package'
                    new_node_parent.add_child(package_node)
                    new_node_parent = package_node
                    node_stack.append(package_node)
                
                # Create the actual module at the leaf
                new_node = ModuleNode(module_name, new_node_parent,
                                  depth=new_node_parent.depth + 1)
                new_node.node_type = 'module'
                new_node_parent.add_child(new_node)
                node_stack.append(new_node)
            else:
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
        new_node = ModuleNode(node_name, current_node, 
                              depth=current_node.depth + 1)
        current_node.add_child(new_node)
        node_stack.append(new_node)

    # POTENTIAL FUTURE IMPLEMENTATIONS
    # TODO: possibly add this in future iterations of the lib, needs more 
    # testing for robustness
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

    node.children = dict((k, v) for (k, v) in node.children.items() \
                         if not v.ignore_package and not v.ignore_module)

    for _, child in node.children.items():
        __flush_pruned_nodes(child)

