


class ParsedCell:
    """
    Base class for parsed cells (purely virtual)
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
    def __init__(self, headers, commands):
        super(ParsedMarkdownCell, self).__init__()
        self.headers = headers
        self.commands = commands  # custom commands added to overwrite module/filename etc.

    def __str__(self):
        parsed_head_str = ''.join(f'\n\t\t\t{h}' for h in self.headers)
        parsed_cmd_str = ''.join(f'\n\t\t\t{cmd}' for cmd in self.commands)
        return f'\n\tParsedMarkdownCell(\n\t\tHeaders: [{parsed_head_str}]\n\t\tCommands: [{parsed_cmd_str}]\n\t)'

    def __repr__(self):
        return str(self)