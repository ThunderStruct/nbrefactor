""" Code and Markdown parsed cell structures
"""

import abc
from . import MarkdownHeader, MarkdownCommand


class ParsedCell(abc.ABC):
    """
    Base class for parsed cells (purely virtual; just for typing purposes)
    """
    def __init__(self):
        pass


class ParsedCodeCell(ParsedCell):
    def __init__(self, cell_idx, raw_source, parsed_source, dependencies, 
                 module_node=None):
        super(ParsedCodeCell, self).__init__()
        self.cell_idx = cell_idx
        self.raw_source = raw_source
        self.parsed_source = parsed_source
        self.dependencies = dependencies
        self.module_node = module_node

    def __str__(self):
        return (
            f'\n\tParsedCodeCell(\n\t\tCell Idx: {self.cell_idx}\n\t\tRaw '
            f'Source: {self.raw_source[:50]}...\n\t\t'
            f'Parsed Source: {self.parsed_source[:50]}...\n\t\Dependencies: '
            f'{self.dependencies}\n\t\tModuleNode: {self.module_node}\n\t)')

    def __repr__(self):
        return str(self)
    

class ParsedMarkdownCell(ParsedCell):
    def __init__(self, elements, warnings=[]):
        super(ParsedMarkdownCell, self).__init__()
        
        self.elements = elements    # a list of both MarkdownHeader and 
                                    # MarkdownCommand objects
        self.warnings = warnings    # a list of dicts containing raised 
                                    # warnings/linting results

    def __str__(self):
        elements_str = ''.join(f'\n\t\t\t{el}' for el in self.elements)
        warnings_str = ''.join(f'\n\t\t\t{w}' for w in self.warnings)

        return (
            f'\n\tParsedMarkdownCell(\n\t\Elements: [{elements_str}],\n\t'
            f'Warnings: [{warnings_str}]\n\t)'
        )

    def __repr__(self):
        return str(self)
    
    @property
    def headers(self):
        return [isinstance(elem, MarkdownHeader) for elem in self.elements]
    
    @property
    def commands(self):
        return [isinstance(elem, MarkdownCommand) for elem in self.elements]
    
    