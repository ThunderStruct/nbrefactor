""" Code Dependency Analyzer and Injector

Used to analyze import statements in the refactoring process,
as well as injecting the dependencies to respective files.
"""

import ast
import astor
from collections import defaultdict

def __get_definitions(code):
    """
    Extract function and class definitions from code.

    Args:
    code (str): Source code.

    Returns:
    list: List of AST nodes representing the function and class definitions.
    """
    tree = ast.parse(code)
    func_and_class_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    return func_and_class_nodes

def __get_imports_from_code(import_lines):
    """
    Parse import statements from code.

    Args:
    import_lines (list): List of import lines.

    Returns:
    list: List of AST nodes representing the import statements.
    """
    combined_imports = '\n'.join(import_lines)
    tree = ast.parse(combined_imports)
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    return import_nodes

def __get_usages(code, defined_names):
    """
    Find usages of defined names (functions, classes) in the code.

    Args:
    code (str): Source code.
    defined_names (set): Set of defined names to track.

    Returns:
    set: Set of used names found in the code.
    """
    tree = ast.parse(code)
    used_names = set()

    class UsageVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id in defined_names:
                used_names.add(node.id)

    UsageVisitor().visit(tree)
    return used_names

def analyze_dependencies(code_cells, import_lines):
    """
    Analyze code cells to determine dependencies.

    Args:
    code_cells (dict): Dictionary of code cells.
    import_lines (list): List of import lines.

    Returns:
    dict: Dictionary mapping module names to their imports and code.
    """
    dependencies = defaultdict(lambda: {'imports': [], 'code': []})
    import_nodes = __get_imports_from_code(import_lines)
    
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

