""" CLI Option Parsing and Handling
"""

import argparse

def get_args():
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

    return parser.parse_args()