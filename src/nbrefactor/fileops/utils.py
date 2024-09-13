"""File utilities' methods used across the library 
"""

import os

def ensure_dir(path):
    """
    Ensures that the given directory exists, and \
        creates it if it does not exist
    
    Args:
        path (str): Path to the dir.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_ipynb_validity(path):
    """
    Ensures that the given path points to a valid .ipynb file.
    
    Args:
        path (str): Path to the Jupyter notebook file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a .ipynb file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f'The file "{path}" does not exist.')
    
    if not path.endswith('.ipynb'):
        raise ValueError(
            f'The file "{path}" is not a valid '
            'Jupyter notebook (.ipynb) file.'
        )
    
    # path exists and is valid
    return path


def compute_plot_path(path, ipynb_fname):
    """
    Computes the full plot path. If the given path is a directory, \
        constructs a plot filename
    using the provided notebook filename and plot extension. Ensures that \
        the directory exists.
    
    Args:
        path (str): Path to a directory or file where the plot should be saved.
        ipynb_fname (str): The name of the Jupyter notebook file \
            (without extension).
    
    Returns:
        str: The full path to the plot
    """

    # ensure/create dir
    ensure_dir(path)

    if os.path.isdir(path):
        # compute default plot filename if the path is a directory 
        # (i.e. w/o filename)
        plot_filename = f'{os.path.splitext(ipynb_fname)[0]}'
        return os.path.join(path, plot_filename)
    
    else:
        # path contains filename, return as is
        return path
