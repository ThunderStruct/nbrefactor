# Current
__version__ = '0.1.6'

# CLI
from .cli import main

# Exposed submodules
from .datastructs import *
from .processor import process_notebook
from .visualization import plot_module_tree