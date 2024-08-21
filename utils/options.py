""" CLI Option Parsing and Handling
"""

import argparse

class Options:

    __ARGS = {}

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='An automation tool to \
                                        refactor Jupyter Notebooks to Python \
                                        modules and vice versa, with dependency \
                                        analysis.')
        
        parser.add_argument('-nb', '--notebook_path', 
                            type=str,
                            dest='notebook_path',
                            help='Path to the Jupyter Notebook.')
        
        parser.add_argument('-out', '--output_path',
                            type=str,
                            dest='output_path',
                            help='Path to the output directory.')
        
        Options.__ARGS = vars(parser.parse_args())


    @staticmethod
    def get(option):
        if option in Options.__ARGS:
            return Options.__ARGS[option]
    
        raise KeyError(f'Invalid option {option}')
    
    @staticmethod
    def get_options():
        return Options.__ARGS