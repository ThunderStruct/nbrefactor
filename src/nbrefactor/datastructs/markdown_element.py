""" Notebook markdown data structures
"""

import abc
from enum import Enum

class MarkdownElement(abc.ABC):
    """
    Base class for markdown elements (purely virtual; just for typing purposes)
    """
    
    def __init__(self):
        pass
    

class MarkdownCommandType(Enum):

    # Node-Skipping Commands
    IGNORE_PACKAGE      = 'ignore-package'      # ignores all modules/packages 
                                                # encountered until a (<= depth) 
                                                # is reached
    IGNORE_MODULE       = 'ignore-module'       # ignores a single module 
                                                # (could consist of multiple 
                                                # code cells)
    IGNORE_CELL         = 'ignore-cell'         # ignores the next code-cell, 
                                                # regardless of its type.
    IGNORE_MARKDOWN     = 'ignore-markdown'     # ignores the current Markdown 
                                                # cell (e.g. when MD is used 
                                                # for instructions only)
    ANALYZE_ONLY        = 'analyze-only'        # analyzes the next code cell for
                                                # dependencies but does not create
                                                # a file for it

    # Node-Manipulation Commands
    RENAME_PACKAGE      = 'package'             # sets the package name 
                                                # (and asserts node type)
    RENAME_MODULE       = 'module'              # sets the module name 
                                                # (and asserts node type)
    RENAME_NODE         = 'node'                # sets the node name 
                                                # generically regardless of 
                                                # node type

    # Node-Declaration Commands all take a string value for the node name
    # and are placed at sibling-depth in the node tree.
    # TODO: could optionally accept tuples of (name, level) for better 
    # customizability,
    # no need to even properly parse this, we could just try-except 
    # type-casting the value.
    # If the project grows, we can maybe define an EBNF grammar for all 
    # the options? overkill?
    DECLARE_PACKAGE     = 'declare-package'     # declares a new node and
                                                # asserts its 'package' type
    DECLARE_MODULE      = 'declare-module'      # declares a new node and 
                                                # asserts its 'module' type
    DECLARE_NODE        = 'declare-node'        # declares a new node, type 
                                                # will be kept as 'None' and #
                                                # inferred accordingly

    # Potential future implementations
    # NODE_DEPTH          = 'node-depth'          # overrides the node's `level` 
    #                                             # (need to figure out the 
    #                                             # necessary constraints here)
    # PREPEND_VALUE       = 'prepend'             # prepends a given string 
    #                                             # value to the next code-cell
    # APPEND_VALUE        = 'append'              # appends a given string value 
    #                                             # to the next code-cell


class MarkdownCommand(MarkdownElement):
    def __init__(self, cmd_str, value):
        try:
            self.type = MarkdownCommandType(cmd_str)    # raises value error 
                                                        # if invalid
        except:
            # re-raise the error with our custom code and message
            raise ValueError({
                'source': 'MarkdownCommand',
                'code': 1,
                'type': 'Invalid Command',
                'subject': cmd_str,
                'message': f'An invalid command "${cmd_str}" was encountered'
            })
        
        if value is not None:
            self.value = value
        else:
            if self.type in [MarkdownCommandType.RENAME_PACKAGE,
                             MarkdownCommandType.RENAME_MODULE,
                             MarkdownCommandType.RENAME_NODE,
                             MarkdownCommandType.DECLARE_PACKAGE,
                             MarkdownCommandType.DECLARE_NODE,
                             MarkdownCommandType.DECLARE_NODE]:
                raise ValueError({
                    'source': 'MarkdownCommand',
                    'code': 2,
                    'type': 'Invalid Value',
                    'subject': cmd_str,
                    'message': f'The "${cmd_str}" command requires a value.'
                })

            self.value = True   # defaults to True 
                                # (i.e. `ignore-cell` is equiv. to 
                                # `ignore-cell=True`)

    # @staticmethod
    # def init_from(cmd_str, value):
    #     """
    #     Factory method to safely instantiate MarkdownCommandType objects 
    #     from strings
    #     """
    #     try:
    #         cmd_type = MarkdownCommandType(cmd_str)
    #         return Command(cmd_type, value)
        
    #     except ValueError:
    #         # invalid command string
    #         print(
    #           f'An invalid command type `{cmd_str}` was found and ignored.'
    #         )
    #         return None

    def __str__(self):
        return f'MarkdownCommand(cmd_type: {self.type}, value: {self.value})'
    
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
    
