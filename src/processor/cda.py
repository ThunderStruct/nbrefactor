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
    __definitions_tracker = {}  # all encountered definitions (handles identifier shadowing as well!)

    def __init__(self, local_module_path):
        self.used_names = set()
        self.local_module_path = local_module_path
        self.required_imports = {}

    def visit_Name(self, node):
        if node.id in UsageVisitor.__definitions_tracker:
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        UsageVisitor.__definitions_tracker[node.name] = {
            'node': node,
            'module_path': self.local_module_path
        }
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        UsageVisitor.__definitions_tracker[node.name] = {
            'node': node,
            'module_path': self.local_module_path
        }
        self.generic_visit(node)

    def handle_import(self, import_node):
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                UsageVisitor.__definitions_tracker[alias.asname or alias.name] = {
                    'node': import_node,
                    'module_path': None  # set None as it's not a local definition
                }
        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ''
            for alias in import_node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                UsageVisitor.__definitions_tracker[alias.asname or alias.name] = {
                    'node': import_node,
                    'module_path': None  # set None as it's not a local definition
                }
        
    def add_import(self, name, module):
        self.required_imports[name] = module

    def get_usages(self):
        return self.used_names

    def get_dependencies(self):
        return [f"from {module} import {name}" if module else f"import {name}" 
                for name, module in self.required_imports.items()]

    @classmethod
    def get_definitions(cls):
        return cls.__definitions_tracker
    

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


def __relative_import_path(current_module, target_module):
    common_prefix_length = 0
    for i in range(min(len(current_module), len(target_module))):
        if current_module[i] == target_module[i]:
            common_prefix_length += 1
        else:
            break

    up_levels = len(current_module) - common_prefix_length
    down_path = target_module[common_prefix_length:]

    return '.' * up_levels + '.'.join(down_path)


def analyze_code_cell(source, current_module_path):

    # 1. clean the source (remove IPython magic statements)
    # (possibly in the future we can parse + run them in a sub-process (i.e. pip installs, etc.))
    clean_source = __remove_ipy_statements(source)

    # 2. parse source
    parsed_tree = ast.parse(clean_source)

    # 3. strip the source of import statements
    imports = []
    code_lines = []
    source_lines = clean_source.splitlines()

    for node in ast.walk(parsed_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif hasattr(node, 'lineno'):
            code_lines.append(source_lines[node.lineno - 1])

    extracted_source = "\n".join(code_lines)

    # 4. track definitions and visit the AST
    usage_visitor = UsageVisitor(current_module_path)

    for import_node in imports:
        usage_visitor.handle_import(import_node)

    usage_visitor.visit(parsed_tree)

    
    # 5. analyze dependencies based on parsed AST usages
    for used_name in usage_visitor.get_usages():
        defs = UsageVisitor.get_definitions()
        if used_name in defs:
            definition_info = defs[used_name]
            module_path = definition_info['module_path']

            # relative import path
            if module_path != current_module_path and module_path is not None:
                rel_path = __relative_import_path(current_module_path, module_path)
                usage_visitor.add_import(used_name, rel_path)
            else:
                # the usage is within the same module, no import needed
                usage_visitor.add_import(used_name, None)

    ret_dict = {
        'source': extracted_source,
        'dependencies': usage_visitor.get_dependencies(),
    }

    print(ret_dict, '\n\n')

    return ret_dict