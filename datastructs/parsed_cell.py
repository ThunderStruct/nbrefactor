
import abc
from . import MarkdownHeader, MarkdownCommand


class ParsedCell(abc.ABC):
    """
    Base class for parsed cells (purely virtual; just for typing purposes)
    """
    def __init__(self):
        pass


class ParsedCodeCell(ParsedCell):
    def __init__(self, cell_idx, raw_source, parsed_source, dependencies, module_node=None):
        super(ParsedCodeCell, self).__init__()
        self.cell_idx = cell_idx
        self.raw_source = raw_source
        self.parsed_source = parsed_source
        self.dependencies = dependencies
        self.module_node = module_node

    def __str__(self):
        return (f'\n\tParsedCodeCell(\n\t\tCell Idx: {self.cell_idx}\n\t\tRaw Source: {self.raw_source[:50]}...\n\t\t'
                f'Parsed Source: {self.parsed_source[:50]}...\n\t\Dependencies: {self.dependencies}\n\t\t'
                f'ModuleNode: {self.module_node}\n\t)')

    def __repr__(self):
        return str(self)
    

class ParsedMarkdownCell(ParsedCell):
    def __init__(self, elements):
        super(ParsedMarkdownCell, self).__init__()
        
        self.elements = elements    # a list of both MarkdownHeader and MarkdownCommand objects

    def __str__(self):
        elements_str = ''.join(f'\n\t\t\t{el}' for el in self.elements)
        return f'\n\tParsedMarkdownCell(\n\t\Directives: [{elements_str}]\n\t)'

    def __repr__(self):
        return str(self)
    
    @property
    def headers(self):
        return [isinstance(elem, MarkdownHeader) for elem in self.elements]
    
    @property
    def commands(self):
        return [isinstance(elem, MarkdownCommand) for elem in self.elements]
    
    