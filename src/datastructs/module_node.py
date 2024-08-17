

from .parsed_cell import ParsedCodeCell

class ModuleNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = {}
        self.parsed_cells = []

    def add_child(self, child_node):
        self.children[child_node.name] = child_node

    def add_parsed_cell(self, parsed_cell):
        self.parsed_cells.append(parsed_cell)

    def get_full_path(self):
        if self.parent:
            return self.parent.get_full_path() + [self.name]
        return [self.name]
    
    def has_code_cells(self):
        return any([True for cell in self.parsed_cells if isinstance(cell, ParsedCodeCell)])

    def __str__(self):
        return f'ModuleNode(name={self.name}, parent={self.parent.name}, children={list(self.children.keys())}, parsed cells={self.parsed_cells})'