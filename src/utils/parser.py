import re
from copy import copy
from .datastructs import MarkdownHeader, Command
from .datastructs import ParsedCodeCell, ParsedMarkdownCell

def parse_notebook_code(cell_idx, source):
    """
    Formats and validates a Notebook code cell (i.e., removing 
    invalid lines - such as iPython magic commands - to prep 
    for refactoring)

    Args:
        cell_idx (int): the current cell's index
        source (str): source code (str lines); the ipynb code cell
    
    Returns:
        (ParsedCodeCell): a parsed code cell object, containing the \
        parsed import statements and formatted code
    """

    def __format_code_line(line):
        if line.startswith('!'):
            # skip iPython magic commands
            return ''
        
        return line

    ret_code = []
    ret_imports = []
    is_multiline_comment = False

    for line in source.split('\n'):
        # parse imports (whilst handling keywords 'from'/'import' in multiline comments)
        stripped_line = line #.strip()
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            is_multiline_comment = not is_multiline_comment
        if not is_multiline_comment:
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                ret_imports.append(line)
            else:
                ret_code.append(__format_code_line(line))
        else:
            ret_code.append(__format_code_line(line))
    
    return ParsedCodeCell(source, ret_imports, '\n'.join(ret_code))


def parse_notebook_markdown(cell_idx, source):
    """
    Formats and validates a Notebook code cell (i.e., removing 
    invalid lines - such as iPython magic commands - to prep 
    for refactoring)

    Args:
        cell_idx (int): the current cell's index
        source (str): source code (str lines); the ipynb code cell
    
    Returns:
        (ParsedMarkdownCell): a parsed markdown cell object containing \
        the parsed MarkdownHeader and Command objects.
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

    return ParsedMarkdownCell(source, md_headers, commands)