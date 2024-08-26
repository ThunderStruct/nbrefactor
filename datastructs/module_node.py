

from .parsed_cell import ParsedCodeCell

class ModuleNode:

    def __init__(self, name, parent=None, depth=0, node_type=None):
        self.name = name
        self.parent = parent
        self.children = {}
        self.parsed_cells = []
        self.depth = depth
        self.node_type = node_type       # ['package', 'module']


    def add_child(self, child_node):
        self.children[child_node.name] = child_node


    def add_parsed_cell(self, parsed_cell):
        self.parsed_cells.append(parsed_cell)


    def get_full_path(self):
        if self.parent:
            return self.parent.get_full_path() + [self.name]
        return [self.name]
    

    def has_code_cells(self):
        """
        Checks if the node has non-empty code cells (we only write modules/.py files if this is true)
        """
        return any([True for cell in self.parsed_cells \
                    if isinstance(cell, ParsedCodeCell) and len(cell.parsed_source.strip())])
    

    def has_children(self):
        return len(self.children) > 0
    

    def is_top_level_module(self):
        # has parent but no grand-parent (i.e. parent is the root)
        return self.parent is not None and self.parent.parent is None
    

    def set_node_type(self, node_type):
        """
        Forces a node type (through a MarkdownCommand)
        """

        self.node_type = node_type


    def aggregate_dependencies(self):
        """
        Gets the union of all `dependencies` sets
        """

        all_dependencies = set()
        for parsed_cell in self.parsed_cells:
            if isinstance(parsed_cell, ParsedCodeCell):
                all_dependencies.update(parsed_cell.dependencies)
        return all_dependencies


    def __str__(self):
        return (
            f'ModuleNode(name={self.name}, type={self.node_type or ""}, '
            f'parent={self.parent.name}, children={list(self.children.keys())}, '
            f'parsed cells={self.parsed_cells})'
        )