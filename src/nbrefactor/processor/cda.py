""" Code Dependency Analyzer

Used to parse code, track declared definitions, and analyze dependencies.
"""

import re
import ast

# [DEPRECATED] - we cannot use the parsed tree to remove IPython magic as the 
# parsing fails due to the existence of magic statements
# class MagicRemover(ast.NodeTransformer):
#     def visit_Expr(self, node):
#         # prune nodes that are strings and match the ipy magic pattern
#         if isinstance(node.value, ast.Constant) and re.match(r'^\%|\!', 
#                       node.value.s):
#             return None
        
#         # prune nodes that are Name or Call that start with % or !
#         elif isinstance(node.value, (ast.Name, ast.Call)) and \
#               hasattr(node.value, 'func') and isinstance(node.value.func, 
#                                                          ast.Name):
#             if re.match(r'^\%|\!', node.value.func.id):
#                 return None
            
#         return node

class UsageVisitor(ast.NodeVisitor):
    """
    An AST visitor class to handle the visited definitions, usages, 
    and dependencies.
    """
    __definitions_tracker = {}  # all encountered definitions

    def __init__(self, local_module_path):
        self.local_module_path = local_module_path
        self.used_names = set()         # usage tracker
        self.local_scope = []           # for scope-awareness (this is used to 
                                        # detect definition shadowing)


    def visit_FunctionDef(self, node):
        
        if len(self.local_scope) == 0:
            # add the function to the global definitions tracker
            # ONLY if it is declared in the global scope
            UsageVisitor.__definitions_tracker[node.name] = {
                'node': node,
                'node_type': 'ImportFrom',
                'module_path': self.local_module_path,
                'is_local': True,
                'alias': None
            }

        # open scope for func (that way nested definitions won't be added)
        self.local_scope.append(set())

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

        # close func scope
        self.local_scope.pop()


    def visit_ClassDef(self, node):
        # open scope
        self.local_scope.append(set())
        
        UsageVisitor.__definitions_tracker[node.name] = {
            'node': node,
            'node_type': 'ImportFrom',
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
            # a new usage in the given parse tree that also exists in the
            # global defs 
            # (must be imported)
            self.used_names.add(node.id)
        self.generic_visit(node)


    def visit_Assign(self, node):
        # handle assignment statements (edge case where they are treated as 
        # global defs)
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
                    'node_type': 'Import',
                    'module_path': alias.name,
                    'is_local': False,
                    'alias': alias.asname
                }

        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ''
            for alias in import_node.names:
                defined_name = alias.asname or alias.name
                full_name = f"{module}" if module else alias.name
                UsageVisitor.__definitions_tracker[defined_name] = {
                    'node': import_node,
                    'node_type': 'ImportFrom',
                    'module_path': full_name,
                    'is_local': False,
                    'alias': alias.asname
                }

    def get_usages(self):
        return self.used_names
    

    def __relative_import_path(self, target_module):
        """
        Builds up the relative import statement from the local module path 
        and the passed target module paths.
        """

        prefix_len = 0
        for i in range(min(len(self.local_module_path), len(target_module))):
            if self.local_module_path[i] == target_module[i]:
                prefix_len += 1
            else:
                break

        up_levels = len(self.local_module_path) - prefix_len
        down_path = target_module[prefix_len:]

        if up_levels > 0:
            return '.' * up_levels + '.'.join(down_path)
        else:
            return '.'.join(down_path)


    def get_dependencies(self):
        # generate import statements to inject the dependencies in the 
        # file-writing phase

        # intersection between the used_names set | definitions
        required_imports = {key: UsageVisitor.__definitions_tracker[key] \
                            for key in self.used_names \
                                if key in UsageVisitor.__definitions_tracker}
        
        dependencies = set()
        package_level_module = f'.{self.local_module_path[-2]}' \
            if len(self.local_module_path) > 1 else ''
        for name, definition in required_imports.items():
            # Debugging
            # print(
            #     f'\n----{self.local_module_path} > {name}:'
            #     f' {definition}----\n'
            # )

            alias = definition['alias']
            module_path = definition['module_path']
            is_local = definition['is_local']
            asname = f' as {alias}' if alias is not None else ''

            if module_path == self.local_module_path:
                # used name was declared in this module; import is n/a
                continue

            if is_local:
                # local/relative import statement
                # adjust relative import path given the current module
                module_path = self.__relative_import_path(module_path)
            
            if module_path == '.':
                # package-level relative module
                module_path = package_level_module

            if definition['node_type'] == 'ImportFrom': # from-import
                dependencies.add(f'from {module_path} import {name}{asname}')

            else: # regular import
                dependencies.add(f'import {module_path}{asname}')

        return dependencies

    @classmethod
    def get_definitions(cls):
        return cls.__definitions_tracker



def __remove_ipy_statements(source):
    """
    Clean the sourec of IPython magic commands.

    TODO: consider replacing the magic commands' lines 
    with a `subprocess.run()` command? I believe the `!` commands 
    would simply be `subprocess.run(command_str.split(' ')` or so, 
    and the `%` commands would be ommitted)
    
    Args:
        source (str): The original source code.
        
    Returns:
        str: The cleaned source code with magic commands removed.
    """
    # # This was a dumb approach, ignore
    # magic_pruned = MagicRemover().visit(parsed_tree)
    # return magic_pruned

    # remove lines that start with % or ! with regex rather than 
    # with the ast transformer
    magic_regex = re.compile(r'^(\s*)([!%][^\s=][^\n]*)$', re.MULTILINE)

    # simply comment the statement out rather than remove it (felt wrong to 
    # just remove it)
    # we also account for indented blocks now (i.e. magic statements as sole 
    # statements in if-blocks could be problemtic, we now add a `pass` 
    # statement prior to the commented match)
    cleaned_source = magic_regex.sub(r'\1pass # \2', source)

    return cleaned_source


def analyze_code_cell(source, current_module_path):
    """
    Analyzes dependencies and tracks declared definitions from a given code 
    block.

    There are a few challenges with this, mainly:

    - Tracking declarations globally to handle identifier shadowing 
      (and exclude non-exportable identifiers, such as function arguments)
    - Implementing scope-awareness (this is obviously pretty simple and yet 
      headache-inducing; we just add an empty set `{}` for each encountered 
      scope and push in the respective declarations)
    - Handling relative import statements sequentially as we encounter new 
      modules/code blocks (e.g. `class A` declared in a generated 
      `./root/package_a/class_a.py` could be imported in a relative 
      `package_b`)
    - Taking into consideration import aliases (`import pandas as pd`, etc.)

    One existential crisis later, I split the code dependency analysis 
    process into 5 steps:

    1. We remove ast-incompatible statements from the given code source 
       (primarily IPython magic statements).
    2. Parse the source using ast (this potentially raises an exception, 
       we wrap this function in a try-except on the parser-level, 
       and simply warn + dump the unparseable code into the respective file).
    3. Strip all import statements from the source as we will later inject 
       the ones we need for this particular module.
    4. Track the import statements' declared definitions globally 
       (this will shadow existing identifiers with the same name, as it 
       should). 
       This handles both `import $` and `from $ import $`, supporting `as` 
       aliases.
    5. Grab the visited usages and collate the required dependencies from 
       the global definitions, handling relative imports and 3rd party 
       libraries respectively.


    Args:
        source (str): the raw source code
        current_module_path (list): a list of the current source code's \
            target path components,
        this is used to assign the definitions declared in the source code \
            to a path 
        (so it can later relatively imported when needed)

    Returns:
        dict: a dict containing the parsed source code (key='source') and a \
            list of 
        dependencies/formatted import statements (key='dependencies').
    """

    # 1. clean the source (remove IPython magic statements)
    # (possibly in the future we can parse + run them in a sub-process 
    # (i.e. pip installs, etc.))
    clean_source = __remove_ipy_statements(source)

    # 2. parse source
    parsed_tree = ast.parse(clean_source)

    # 3. strip the source of import statements 
    # (step 4. updates the definitions accordingly)
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
    dependencies = usage_visitor.get_dependencies()

    # [DEPRECATED - the block below was simplified 
    # and consolidated into `UsageVisitor.get_dependencies()`]

    # for used_name in usage_visitor.get_usages():

        # defs = UsageVisitor.get_definitions()
        # if used_name in defs:
        #     definition_info = defs[used_name]

            # module_path = definition_info['module_path'] or None

            # if definition_info['is_local'] \
            #     and module_path != current_module_path:
            #     # relative import
            #     usage_visitor.add_import(used_name, 
            #                              rel_path, 
            #                              definition_info['alias'])

            # elif module_path != current_module_path:
            #     # regular lib import
            #     usage_visitor.add_import(used_name, module_path, 
            #                              definition_info['alias'])

            # # else:
            # #     # the usage is within the same module, no import needed

    return {
        'source': extracted_source,
        'dependencies': dependencies,
    }

