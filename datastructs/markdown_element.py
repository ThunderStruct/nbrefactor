
import abc
from enum import Enum

class MarkdownElement(abc.ABC):
    """
    Base class for markdown elements (purely virtual; just for typing purposes)
    """
    
    def __init__(self):
        pass
    

class MarkdownCommandType(Enum):

    IGNORE_PACKAGE      = 'ignore-package'
    IGNORE_MODULE       = 'ignore-module'
    IGNORE_CELL         = 'ignore-cell'

    NODE_NAME           = 'node-name'
    # NODE_DEPTH          = 'node-depth'


class MarkdownCommand(MarkdownElement):
    def __init__(self, cmd_str, value):
        self.type = MarkdownCommandType(cmd_str)    # raises value error if invalid

        if value is not None:
            self.value = value
        else:
            assert self.type != MarkdownCommandType.NODE_NAME, 'You must provide a value for a NODE_NAME command'

            self.value = True   # defaults to True 
                                # (i.e. `ignore-cell` is equiv. to 
                                # `ignore-cell=True`)

    # @staticmethod
    # def init_from(cmd_str, value):
    #     """
    #     Factory method to safely instantiate MarkdownCommandType objects from strings
    #     """
    #     try:
    #         cmd_type = MarkdownCommandType(cmd_str)
    #         return Command(cmd_type, value)
        
    #     except ValueError:
    #         # invalid command string
    #         print(f'An invalid command type `{cmd_str}` was found and ignored.')
    #         return None

    def __str__(self):
        return f'MarkdownCommandType(cmd_type: {self.type}, value: {self.value})'
    
    def __repr__(self):
        return str(self)
    

class MarkdownHeader(MarkdownElement):

    def __init__(self, name, level):

        self.name = name
        self.level = level


    def __str__(self):
        return f'MarkdownHeader(Name: "{self.name}", Level: {self.level})'
    
    def __repr__(self):
        return str(self)
    
