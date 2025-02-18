""" Node representation used to construct the refactored module tree
"""

from .parsed_cell import ParsedCodeCell

class ModuleNode:

    def __init__(self, name, parent=None, depth=0, node_type=None):
        self.name = name                    # node's name
        self.parent = parent                # node's parent (this is `None` in 
                                            # the case of the root)
        self.children = {}                  # node's children (forms the tree 
                                            # structure)
        self.parsed_cells = []              # ParsedCell objects associated 
                                            # with this node
        self.depth = depth                  # node's depth in the tree
        self.node_type = node_type          # options: ['package', 'module']

        # MarkdownCommandType ignore commands 
        # (ignored in the processing phase; will not be parsed or tracked in 
        # the CDA)
        self.ignore_package = False         # prunes the entire branch 
                                            # (i.e package) from the tree
        self.ignore_module = False          # ignores all upcoming parsed cells 
                                            # under this node
        self.ignore_next_cell = False       # ignores the next parsed cell 
                                            # (handled intrinsically)
        self.analyze_only = False           # analyze but don't create a file


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
        Checks if the node has non-empty code cells (we only write \
            modules/.py files if this is true)
        """
        # Don't count this node as having code cells if it's analyze_only
        if self.analyze_only:
            return False
        return any([True for cell in self.parsed_cells \
                    if isinstance(cell, ParsedCodeCell) \
                        and len(cell.parsed_source.strip())])
    

    def has_children(self):
        return len(self.children) > 0
    

    def is_top_level_module(self):
        # has parent but no grand-parent (i.e. parent is the root)
        return self.parent is not None and self.parent.parent is None


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
            f'parent={self.parent.name}, '
            f'children={list(self.children.keys())}, '
            f'parsed cells={self.parsed_cells})'
        )