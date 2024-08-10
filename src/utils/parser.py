import re
from copy import copy

# ^\$(?P<command>\w+(?:-\w+)*)(?:=(?P<value>.*))?$   cmds parse
# <!--(?P<comment>(?:(.|\n)*))-->

def parse_notebook_code(source):
    """
    Formats and validates a Notebook code cell (i.e., removing 
    invalid lines - such as iPython magic commands - to prep 
    for refactoring)

    Args:
        source (str): source code (str lines); the ipynb code cell
    
    ReturnsL
        (str): the formatted code cell
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
    
    return ret_imports, '\n'.join(ret_code)


def parse_notebook_markdown(source, current_hierarchy):
    
    md_str = copy(source)

    # html multiline comment pattern
    comment_regex = re.compile(r'<!--(?P<comment>(?:(.|\n)*?))-->')

    comments = []
    for match in comment_regex.finditer(md_str):
        comments.append(match.group('comment'))

        # removing the html comments from the MD
        md_str = md_str.replace(match.group(0), '')

    cmd_regex = re.compile(r'^\$(?P<command>\w+(?:-\w+)*)(?:=(?P<value>.*))?$', re.MULTILINE)

    for comment in comments:
        for line in comment.split('\n'):
            # strip leading/trailing whitespaces for the cmd=val pattern
            for cmd_match in cmd_regex.finditer(line.strip()):
                command = cmd_match.group('command')
                value = cmd_match.group('value')
                print(f"Command: {command}, Value: {value}")

    
    # handle markdown (md_str is stripped of html comments)
    
    for line in md_str.split('\n'):
        if line.startswith('#'):
            level = line.count('#')
            header = line.strip('#').strip().replace(' ', '_').lower()
            
            if level <= len(current_hierarchy):
                current_hierarchy = current_hierarchy[:level-1]
            current_hierarchy.append(header)