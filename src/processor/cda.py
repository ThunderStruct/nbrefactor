""" Code Dependency Analyzer

Used to parse code, track declared definitions, and analyze dependencies.
"""

import re
import ast

# [DEPRECATED] - we cannot use the parsed tree to remove IPython magic as the parsing fails due to the existence of magic statements
# class MagicRemover(ast.NodeTransformer):
#     def visit_Expr(self, node):
#         # prune nodes that are strings and match the ipy magic pattern
#         if isinstance(node.value, ast.Constant) and re.match(r'^\%|\!', node.value.s):
#             return None
        
#         # prune nodes that are Name or Call that start with % or !
#         elif isinstance(node.value, (ast.Name, ast.Call)) and hasattr(node.value, 'func') and isinstance(node.value.func, ast.Name):
#             if re.match(r'^\%|\!', node.value.func.id):
#                 return None
            
#         return node
    
class UsageVisitor(ast.NodeVisitor):
    def __init__(self, definitions):
        self.used_names = set()
        self.definitions = definitions

    def visit_Name(self, node):
        if node.id in self.definitions:
            self.used_names.add(node.id)
        self.generic_visit(node)



def __remove_ipy_statements(source):
    """
    Clean the sourec of IPython magic commands.
    
    Args:
        source (str): The original source code.
        
    Returns:
        str: The cleaned source code with magic commands removed.
    """

    # magic_pruned = MagicRemover().visit(parsed_tree)
    # return magic_pruned


    # remove lines that start with % or ! with regex rather than ast transformer
    magic_regex = re.compile(r'^\s*(%|\!).*$', re.MULTILINE)
    cleaned_source = magic_regex.sub('# Removed IPython magic command', source)
    return cleaned_source


def analyze_code_cell(source, definitions):

    # 1. clean the source (remove IPython magic statements)
    # (possibly in the future we can parse + run them in a sub-process (i.e. pip installs, etc.))
    clean_source = __remove_ipy_statements(source)

    # 2. parse source
    parsed_tree = ast.parse(clean_source)

    # 3. extract import statements to be injected respectively later
    imports = [node for node in ast.walk(parsed_tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    # 4. track declared definitions (add newly encountered defs and overwrite/shadow existing definitions)
    new_defs = [node for node in ast.walk(parsed_tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    for node in new_defs:
        definitions[node.name] = node

    # 5. analyze dependencies
    usages = UsageVisitor(definitions)
    usages.visit(parsed_tree)

    
    ret_dict = {
        'imports': imports,
        'definitions': definitions,
        'usages': usages.used_names
    }

    print(ret_dict, '\n\n')

    return ret_dict