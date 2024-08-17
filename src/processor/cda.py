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
    __definitions_tracker = {}  # all encountered definitions

    def __init__(self, local_module_path):
        self.local_module_path = local_module_path
        self.used_names = set()         # usage tracker
        self.required_imports = {}      # dependencies
        self.local_scope = []           # for scope-awareness (this is used to detect definition shadowing)


    def visit_FunctionDef(self, node):       
         # open scope
        self.local_scope.append(set())
        
        # add the function to the global definitions tracker
        UsageVisitor.__definitions_tracker[node.name] = {
            'node': node,
            'module_path': self.local_module_path,
            'is_local': True,
            'alias': None
        }

        # add args to the "ignore tracking" list
        for arg in node.args.args:
            self.local_scope[-1].add(arg.arg)
        
        # python 3.8+ edge case for positional-only args
        if hasattr(node.args, 'posonlyargs'):
            for arg in node.args.posonlyargs:
                self.local_scope[-1].add(arg.arg)

        for arg in node.args.kwonlyargs:
            self.local_scope[-1].add(arg.arg)
        
        if node.args.vararg:
            self.local_scope[-1].add(node.args.vararg.arg)
        if node.args.kwarg:
            self.local_scope[-1].add(node.args.kwarg.arg)

        self.generic_visit(node)

        # close scope
        self.local_scope.pop()


    def visit_ClassDef(self, node):
        # open scope
        self.local_scope.append(set())
        
        UsageVisitor.__definitions_tracker[node.name] = {
            'node': node,
            'module_path': self.local_module_path,
            'is_local': True,
            'alias': None
        }
        self.generic_visit(node)

        # local scope
        self.local_scope.pop()

    def visit_Name(self, node):
        
        if any(node.id in scope for scope in self.local_scope):
            # local variable, ignore
            pass
        elif node.id in UsageVisitor.__definitions_tracker:
            # a new global definition
            self.used_names.add(node.id)
        self.generic_visit(node)


    def visit_Assign(self, node):
        # handle assignment statements (edge case where they are treated as global defs)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if self.local_scope:
                    self.local_scope[-1].add(target.id)
        self.generic_visit(node)

    def handle_import(self, import_node):
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                defined_name = alias.asname or alias.name
                UsageVisitor.__definitions_tracker[defined_name] = {
                    'node': import_node,
                    'module_path': alias.name,
                    'is_local': False,
                    'alias': alias.asname
                }

        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ''
            for alias in import_node.names:
                defined_name = alias.asname or alias.name
                full_name = f"{module}.{alias.name}" if module else alias.name
                UsageVisitor.__definitions_tracker[defined_name] = {
                    'node': import_node,
                    'module_path': full_name,
                    'is_local': False,
                    'alias': alias.asname
                }

    def add_import(self, name, module, alias):
        self.required_imports[name] = (module, alias)

    def get_usages(self):
        return self.used_names

    def get_dependencies(self):
        # generate import statements to later inject dependencies
        dependencies = []
        package_level_module = self.local_module_path[-2] if len(self.local_module_path) > 1 else ''
        for name, (module, alias) in self.required_imports.items():
            # Debugging
            # print(f'\n----{self.local_module_path} > {name}: {module}----\n')

            asname = f' as {alias}' if alias is not None else ''

            if module == '.':
                # package-level module (relative import)
                base_module = f'.{package_level_module}'
                dependencies.append(f'from {base_module} import {name}{asname}')
            elif '.' in module: # from-import
                base_module, imported_name = module.rsplit('.', 1)
                dependencies.append(f'from {base_module} import {imported_name}{asname}')
            
            else: # regular import
                dependencies.append(f'import {module}{asname}')

        return dependencies

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

    # add a comment to indicate the removal (felt wrong not to...)
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

    # 3. strip the source of import statements (step 4. updates the definitions accordingly)
    imports = []
    import_lines = set()

    for node in ast.walk(parsed_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
            import_lines.add(node.lineno)

    extracted_source = '\n'.join(
        line for i, line in enumerate(clean_source.splitlines(), start=1)
        if i not in import_lines
    )

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
            module_path = definition_info['module_path'] or None
            if used_name == 'np':
                print(used_name, module_path, definition_info, current_module_path)

            if definition_info['is_local'] and module_path != current_module_path:
                # relative import
                rel_path = __relative_import_path(current_module_path, module_path)
                usage_visitor.add_import(used_name, rel_path, definition_info['alias'])

            elif module_path != current_module_path:
                # regular lib import
                usage_visitor.add_import(used_name, module_path, definition_info['alias'])

            # else:
            #     # the usage is within the same module, no import needed
            

    print(f'Used names for {".".join(current_module_path)}={usage_visitor.get_usages()}, dependencies={usage_visitor.get_dependencies()}\n')
    return {
        'source': extracted_source,
        'dependencies': usage_visitor.get_dependencies(),
    }

