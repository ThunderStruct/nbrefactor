""" Code Dependency Analyzer

Used to parse code and analyze dependencies.
"""

import re
import ast
import astor
from collections import defaultdict

class MagicRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        # remove nodes that are strings and match the ipy magic pattern
        if isinstance(node.value, ast.Constant) and re.match(r'^\%|\!', node.value.s):
            # remove
            return None
        
        # remove nodes that are Name or Call that start with % or !
        elif isinstance(node.value, (ast.Name, ast.Call)) and isinstance(node.value.func, ast.Name):
            if re.match(r'^\%|\!', node.value.func.id):
                # remove
                return None
            
        return node

class UsageVisitor(ast.NodeVisitor):
    def __init__(self, definitions):
        self.used_names = set()
        self.definitions = definitions

    def visit_Name(self, node):
        if node.id in self.definitions:
            self.used_names.add(node.id)

        return self.used_names
    
class CodeDependencyAnalyzer:

    __DEFINITIONS = {}
    __IMPORTS = {}

        
    def __get_parse_tree(source):
        
        return ast.parse(source)

    def __remove_ipython_magic(parsed_tree):
        """
        
        """
        tree = MagicRemover().visit(parsed_tree)

        return tree

    def __get_definitions(parsed_tree):
        """
        Extract function and class definitions from code.

        Args:
        code (str): Source code.

        Returns:
        list: List of AST nodes representing the function and class definitions.
        """

        return [node for node in ast.walk(parsed_tree) \
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))]


    def __get_imports(parsed_tree):
        """
        Parse import statements from code.

        Args:

        Returns:
        list: List of AST nodes representing the import statements.
        """
        return [node for node in ast.walk(parsed_tree) \
                if isinstance(node, (ast.Import, ast.ImportFrom))]


    def __get_usages(parsed_tree, defined_names):
        """
        Find usages of defined names (functions, classes) in the code.

        Args:
        code (str): Source code.
        defined_names (set): Set of defined names to track.

        Returns:
        set: Set of used names found in the code.
        """

        usages = UsageVisitor(defined_names).visit(parsed_tree)

        return usages


    def analyze_dependencies(code_cells, import_lines):
        """
        Analyze code cells to determine dependencies.

        Args:
        code_cells (list): list of code cells.
        import_lines (list): list of import lines.

        Returns:
        list: list of ParsedCodeCell objects, containing the parsed code cells
        """

        dependencies = defaultdict(lambda: {'imports': [], 'code': []})
        import_nodes = __get_imports(import_lines)
        
        defined_names = set()
        cell_code = {}

        # Collect all defined names and corresponding code
        for module, cells in code_cells.items():
            for code in cells:
                all_nodes = __get_definitions(code)
                for node in all_nodes:
                    defined_names.add(node.name)
                cell_code[module] = cell_code.get(module, '') + '\n' + code

        # Track and resolve dependencies
        for module, code in cell_code.items():
            used_names = __get_usages(code, defined_names)
            dependencies[module]['imports'].extend(import_nodes)
            dependencies[module]['code'].append(code)

            # Include dependencies from other modules
            for used_name in used_names:
                for dep_module, dep_code in cell_code.items():
                    if used_name in dep_code and dep_module != module:
                        all_nodes = __get_definitions(dep_code)
                        for node in all_nodes:
                            if node.name == used_name:
                                dependencies[module]['code'].append(astor.to_source(node))

        # Ensure unique import statements and code
        for module, dep in dependencies.items():
            dep['imports'] = list(set(dep['imports']))
            dep['code'] = list(set(dep['code']))

        return dependencies

