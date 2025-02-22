""" CLI entry-point
"""

import os

from .utils import Logger
from .utils import parse_args
from .fileops import compute_plot_path
from .processor import process_notebook
from .visualization import plot_module_tree
# from .visualization.plot_module_tree import plot_module_tree_nx

def main():
    # get arguments
    args = parse_args()

    # refactoring
    root_node = process_notebook(notebook_path=args.notebook_path, 
                                 output_path=args.output_path, 
                                 root_package=args.root_package,
                                 generate_init=args.generate_init)
    
    # plotting (if applicable)
    if args.generate_plot:
        dag = plot_module_tree(root_node, args.plot_format)
        # plot_module_tree_nx(root_node, './plots/nx_plot_test.pdf')

        plot_path = compute_plot_path(args.plot_path,
                                      os.path.basename(args.notebook_path))
        # render to file
        dag.render(plot_path, cleanup=True)

        Logger.log((
            f'Module tree plot saved to "{plot_path}.{args.plot_format}"'
        ), tag='\nSUCCESS', color=Logger.Color.GREEN)
    
    Logger.horizontal_separator(color=Logger.Color.GREEN)


if __name__ == '__main__':
    main()

