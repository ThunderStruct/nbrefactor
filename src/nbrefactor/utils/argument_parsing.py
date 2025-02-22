""" Argument Parsing and Handling
"""

import argparse

def parse_args():
    """
    CLI argument parser.

    This function defines the following arguments:
    
    Positional arguments:
        notebook_path (str): path to the Jupyter Notebook file.
        output_path (str): path to the output directory where refactored \
            modules will be saved.
    
    Optional arguments:
        -rp, --root-package (str): name of the root package. Defaults to the \
            current directory (".").
        -gp, --generate-plot: generate a plot of the module dependency tree. \
            Defaults to `False`.
        -pf, --plot-format (str): format of the plot (e.g., "pdf", "png"). \
            Defaults to "pdf".
        -i, --init-files: generate __init__.py files in package directories. \
            Defaults to `False`.
    
    Returns:
        argparse.Namespace: parsed arguments from the command line.
    """

    parser = argparse.ArgumentParser(description='An automation tool to \
                                     refactor Jupyter Notebooks to Python \
                                     packages and modules, with dependency \
                                     analysis.')
    
    parser.add_argument('notebook_path', 
                        type=str,
                        help='Path to the Jupyter Notebook.')
    
    parser.add_argument('output_path',
                        type=str,
                        help='Path to the output directory.')
    
    parser.add_argument('-rp', '--root-package', 
                        type=str, 
                        default='.',
                        dest='root_package',
                        help=('Name of the root package. Defaults to ".";' 
                              'i.e. current directory'))
    
    parser.add_argument('-gp', '--generate-plot', 
                        action='store_true', 
                        dest='generate_plot',
                        help='Generate a plot of the module dependency tree.')
    
    parser.add_argument('-pp', '--plot-path',
                        type=str, 
                        default='.',
                        dest='plot_path',
                        help=('Path to the plot\'s output directory. '
                              'Defaults to "." (current dir).'))
    
    parser.add_argument('-pf', '--plot-format',
                        type=str, 
                        default='pdf',
                        dest='plot_format',
                        help=('Format of the plot (e.g., "pdf", "png"). '
                              'Defaults to "pdf".'))

    parser.add_argument('-i', '--init-files',
                        action='store_true',
                        dest='generate_init',
                        help='Generate __init__.py files in package directories.')

    return parser.parse_args()
