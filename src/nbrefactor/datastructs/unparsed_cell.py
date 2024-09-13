""" Generic cell structure used to hold data during the 
notebook-reading phase
"""

class UnparsedCell:
    def __init__(self, cell_idx, cell_type, raw_source):
        self.cell_type = cell_type
        self.cell_idx = cell_idx
        self.raw_source = raw_source

    def __str__(self):
        return (
            f'\n\tUnparsedCell(\n\t\tCell Idx: {self.cell_idx}]\n\t\tCell '
            f'Type: {self.cell_type}\n\t\tSource: {self.raw_source[:50]}\n\t)'
        )
    
    def __repr__(self):
        return str(self)
