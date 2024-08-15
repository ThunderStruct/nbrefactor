


class ParsedCell:
    """
    Base class for parsed cells (purely virtual)
    """
    def __init__(self):
        pass


class ParsedCodeCell(ParsedCell):
    def __init__(self, cell_idx, source, imports=None, definitions=None, usages=None):
        super(ParsedCodeCell, self).__init__()
        self.cell_idx = cell_idx
        self.source = source
        self.imports = imports if imports is not None else []
        self.definitions = definitions if definitions is not None else []
        self.usages = usages if usages is not None else []

    def __str__(self):
        return f'\n\tParsedCodeCell(\n\t\tCell Idx: {self.cell_idx}\n\t\tSource: {self.source[:50]}\n\t\tImports: {self.imports}\n\t\tDefinitions: {self.definitions}\n\t\tUsages: {self.usages}\n\t)'

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