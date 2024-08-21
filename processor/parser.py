
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

    # html multiline comment pattern
    comment_regex = re.compile(r'<!--(?P<comment>(?:(.|\n)*?))-->')

    comments = []
    for match in comment_regex.finditer(md_str):
        comments.append(match.group('comment'))

        # removing the html comments from the MD
        md_str = md_str.replace(match.group(0), '')

    # handle markdown / module hierarchy (md_str is stripped of html comments)
    md_headers = []
    for line in md_str.split('\n'):
        if line.startswith('#'):
            md_level = line.count('#')
            md_header = line.strip('#').strip()

            md_headers.append(MarkdownHeader(md_header, md_level))
    
    # cmds pattern
    cmd_regex = re.compile(r'^\$(?P<command>\w+(?:-\w+)*)(?:=(?P<value>.*))?$', re.MULTILINE)

    commands = []
    for comment in comments:
        for line in comment.split('\n'):
            # strip leading/trailing whitespaces for the cmd=val pattern
            for cmd_match in cmd_regex.finditer(line.strip()):
                cmd_str = cmd_match.group('command')
                value = cmd_match.group('value')
                
                try:
                    commands.append(Command(cmd_str, value))
                except ValueError:
                    # invalid command, warn and ignore
                    print(f'An invalid command type `{cmd_str}` in Cell {cell_idx} was found and ignored. (Headers encountered: "{md_header}")')

    return ParsedMarkdownCell(md_headers, commands)