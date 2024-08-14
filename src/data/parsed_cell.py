

class ParsedCell:
    def __init__(self, raw_source):
        self.raw_source = raw_source

class ParsedCodeCell(ParsedCell):
    def __init__(self, raw_source, imports, parsed_code):
        super(ParsedCodeCell, self).__init__(raw_source)

        self.imports = imports
        self.parsed_code = parsed_code

    def __str__(self):
        parsed_imp_str = ''
        for imp in self.imports:
            parsed_imp_str += '\n\t\t\t' + str(imp)
            
        parsed_code_str = '"' + self.parsed_code.replace('\n', '\\n')[:50] + '..."'
        return f'\n\tParsedCodeCell(\n\t\tImports: [{parsed_imp_str}]\n\t\tParsed Code: {parsed_code_str}\n\t)'
    
    def __repr__(self):
        return str(self)


class ParsedMarkdownCell(ParsedCell):
    def __init__(self, raw_source, headers, commands):
        super(ParsedMarkdownCell, self).__init__(raw_source)

        self.headers = headers
        self.commands = commands

    def __str__(self):
        parsed_head_str = ''
        for h in self.headers:
            parsed_head_str += '\n\t\t\t' + str(h)
        
        parsed_cmd_str = ''
        for cmd in self.commands:
            parsed_cmd_str += '\n\t\t\t' + str(cmd)

        return f'\n\tParsedMarkdownCell(\n\t\tHeaders: [{parsed_head_str}]\n\t\tCommands: [{parsed_cmd_str}]\n\t)'
    
    def __repr__(self):
        return str(self)