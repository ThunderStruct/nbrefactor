from enum import Enum, auto

class CommandType(Enum):

    IGNORE_CELL         = 'ignore-cell'
    IGNORE_MODULE       = 'ignore-module'

    DECLARE_FILE        = 'file'
    DECLARE_MODULE      = 'module'

    ROOT_LEVEL          = 'root-level'


class Command:
    def __init__(self, cmd_str, value):
        self.type = CommandType(cmd_str)    # raises value error if invalid

        if value is not None:
            self.value = value
        else:
            self.value = True   # defaults to True 
                                # (i.e. `ignore-cell` is equiv. to 
                                # `ignore-cell=True`)

    # @staticmethod
    # def init_from(cmd_str, value):
    #     """
    #     Factory method to safely instantiate CommandType objects from strings
    #     """
    #     try:
    #         cmd_type = CommandType(cmd_str)
    #         return Command(cmd_type, value)
        
    #     except ValueError:
    #         # invalid command string
    #         print(f'An invalid command type `{cmd_str}` was found and ignored.')
    #         return None

    def __str__(self):
        return f'Command(cmd_type: {self.type}, value: {self.value})'
    
    def __repr__(self):
        return str(self)