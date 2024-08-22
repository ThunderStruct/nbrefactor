
import sys
sys.path.append('..')

import re
from copy import copy
from datastructs import MarkdownHeader, Command
from datastructs import ParsedCodeCell, ParsedMarkdownCell
from .cda import analyze_code_cell


def parse_code_cell(cell_idx, source, module_node):
    """
    Parses a code cell into a ParsedCodeCell object.
    
    Args:
        cell_idx (int): index of the cell in the notebook.
        source (str): the cell's index in the notebook.
        module_node (:class:`~ModuleNode`): the module node in the built module tree
        
    Returns:
        ParsedCodeCell: a parsed code cell object with analyzed imports, definitions, usages, etc.
    """

    parsed_code = analyze_code_cell(source, module_node.get_full_path())

    return ParsedCodeCell(
        cell_idx=cell_idx,
        raw_source=source,
        parsed_source=parsed_code['source'], 
        dependencies=parsed_code['dependencies'],
        module_node=module_node
    )


def parse_markdown_cell(cell_idx, source):
    """
    Parses a markdown cell into a ParsedMarkdownCell object.
    
    Args:
        cell_idx (int): the cell's index in the notebook.
        source (str): the cell's markdown content.
        
    Returns:
        ParsedMarkdownCell: A parsed markdown cell object containing headers and commands.
    """

    md_str = copy(source)   # copied to prevent in-place modification

    comment_regex = re.compile(r'<!--(?P<comment>(?:(.|\n)*?))-->')
    cmd_regex = re.compile(r'^\$(?P<command>\w+(?:-\w+)*)(?:=(?P<value>.*))?$', re.MULTILINE)

    md_elements = []        # both the Command + MarkdownHeader objects, in the order they appear in
                            # we don't simlpy match and extract commands/headers across the raw source
                            # to maintain the order of execution (in case multiple 
                            # headers/commands are present)


    # line by line parsing (to maintain sequential order)
    for line in md_str.split('\n'):
        clean_line = line.strip()

        # HTML comments' matching (potentially Command objects)
        comment_match = comment_regex.search(clean_line)
        if comment_match:
            comment = comment_match.group('comment')
            comment_lines = comment.split('\n')

            for comment_line in comment_lines:
                # parse comment lines (we're allowing multiple commands 
                # in a single comment block)

                cmd_match = cmd_regex.match(comment_line.strip())
                if cmd_match:
                    cmd_str = cmd_match.group('command')
                    value = cmd_match.group('value')
                    
                    try:
                        md_elements.append(Command(cmd_str, value))
                    except ValueError:
                        print((
                            f'WARNING: An invalid command type `{cmd_str}` in'
                            f' Cell #{cell_idx} was found and ignored.'
                        ))

            continue

        # MD header -> extract node depth/level and name of package/module
        if clean_line.startswith('#'):
            md_level = clean_line.count('#')
            md_header = clean_line.strip('#').strip()
            md_elements.append(MarkdownHeader(md_header, md_level))

    return ParsedMarkdownCell(md_elements)


