
class MarkdownHeader:

    def __init__(self, name, level):

        self.name = name
        self.level = level


    def __str__(self):
        return f'MarkdownHeader(Name: "{self.name}", Level: {self.level})'
    
    def __repr__(self):
        return str(self)