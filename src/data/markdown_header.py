
class MarkdownHeader:

    def __init__(self, header, level):

        self.header = header
        self.level = level


    def __str__(self):
        return f'MarkdownHeader(Header: "{self.header}", Level: {self.level})'
    
    def __repr__(self):
        return str(self)