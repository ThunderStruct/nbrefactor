import pandas as pd
import yaml
import errno
from .prompt_handler import PromptHandler
import os
import pickle
"""HiveNAS file-handling methods
"""

class FileHandler:
    '''Wrapper for file-handling methods
    '''

    __VALID_PATHS = {}

    @staticmethod
    def __path_exists(path):
        '''Checks if file exists

        Args:
            path (str): path to file (includes filename and extension)

        Returns:
            bool: whether or not the file exists
        '''

        return os.path.exists(path)


    @staticmethod
    def validate_path(path):
        '''Ensures that a given directory path is universaly valid \
        (Windows/Linux/MacOS/POSIX) and creates it.

        Prompts user for overwriting (using :class:`~utils.prompt_handler.PromptHandler`) \
        if it already exists

        Args:
            path (str): path to validated

        Returns:
            bool: validity of the given path
        '''

        # Directory already exists, prompt for overwrite permission (first time only)
        if path not in FileHandler.__VALID_PATHS and FileHandler.__path_exists(path):

            FileHandler.__VALID_PATHS[path] = True

            if len(os.listdir(path)) == 0:
                # directory exists and is empty -> is valid
                return FileHandler.__VALID_PATHS[path]

            # directory exists and is NOT empty
            FileHandler.__VALID_PATHS[path] = True
            print(f'\nPath ({path}) already exists!\n\n')
            return PromptHandler.prompt_yes_no('Would you like to overwrite files in this path?')

        # Previously evaluated
        if path in FileHandler.__VALID_PATHS:
            return FileHandler.__VALID_PATHS[path]

        # Check the validity of the given path
        try:
            os.makedirs(path)
            FileHandler.__VALID_PATHS[path] = True
        except OSError as e:
            # path invalid
            print('\nBase path and/or config version are invalid! Please choose path-friendly names.\n')
            FileHandler.__VALID_PATHS[path] = False

        return FileHandler.__VALID_PATHS[path]


    @staticmethod
    def create_dir(path):
        '''Recursively creates new directory if it does not exist

        Args:
            path (str): directory path to be created
        '''

        if not FileHandler.__path_exists(path):
            os.makedirs(path)


    @staticmethod
    def path_must_exist(path):
        '''Checks if file exists and raises error if it is not.
        Used when the logic of the algorithm depends on the loaded file

        Args:
            path (str): path to file

        Raises:
            :class:`FileNotFoundError`: file does not exist
        '''

        if not FileHandler.__path_exists(path):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    path)


    @staticmethod
    def save_pickle(p_dict, path, filename, force_dir=True):
        '''Saves the given dictionary as a :class:`pickle`

        Args:
            p_dict (dict): data to be saved
            path (str): path to save directory
            filename (str): output filename
            force_dir (bool, optional): whether or not to force create \
            the directory if it does not exist

        Returns:
            bool: save operation status
        '''

        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                # directory does not exist and cannot create dir
                return False

        # dump pickle
        with open(os.path.join(path, filename), 'wb') as handle:
            pickle.dump(p_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return True


    @staticmethod
    def load_pickle(path, default_dict={}):
        '''Loads :class:`pickle`  and returns decoded dictionary

        Args:
            path (str): path to :class:`pickle` file (includes filename)
            default_dict (dict, optional): default dictionary to return \
            if the pickle does not exist (defaults to :code:`{ }`)

        Returns:
            dict: loaded data
        '''

        res = default_dict

        if FileHandler.__path_exists(path):
            with open(path, 'rb') as handle:
                res = pickle.load(handle)

        return res


    @staticmethod
    def save_df(df, path, filename, force_dir=True):
        '''Saves a Pandas DataFrame to the given path

        Args:
            df (:class:`pandas.DataFrame`): dataframe to be saved
            path (str): save directory path
            filename (str): output filename
            force_dir (bool, optional): whether or not to force create \
            the directory if it does not exist

        Returns:
            bool: save operation status
        '''

        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                # directory does not exist and cannot create dir
                return False

        # save dataframe
        df.to_csv(os.path.join(path, filename))


    @staticmethod
    def load_df(path, default_df=None):
        '''Loads a :class:`pandas.DataFrame`

        Args:
            path (str): path to the dataframe
            default_df (None, optional): default dictionary to return \
            if the dataframe does not exist (defaults to empty dataframe)

        Returns:
            :class:`pandas.DataFrame`: loaded dataframe or default
        '''

        res = default_df or pd.DataFrame()

        if FileHandler.__path_exists(path):
            res = pd.read_csv(path, header=0, index_col=0)

        return res


    @staticmethod
    def export_yaml(config_dict, path, filename, file_version_comment='', force_dir=True):
        '''Exports a given dictionary to a yaml file

        Args:
            config_dict (dict): dictionary to be saved as yaml
            path (str): save directory path
            filename (str): output filename
            file_version_comment (str, optional): optional string to be prepended at /
            the top of the yaml file as a comment (typically used to highlight the /
            configuration version)
            force_dir (bool, optional): whether or not to force create \
            the directory if it does not exist

        Returns:
            bool: save operation status
        '''

        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                return False

        with open(os.path.join(path, filename), 'w') as handler:
            if file_version_comment:
                handler.write(f'\n# {file_version_comment}\n\n')
            yaml.dump(config_dict, handler, default_flow_style=False)

        return True


    @staticmethod
    def load_yaml(path, loader, default_dict={}):
        '''Loads a yaml config file and returns it as dict

        Args:
            path (str): path to the yaml file (includes filename)
            loader (:class:`yaml.SafeLoader`): yaml custom loader defined \
            in :func:`~config.params.Params.export_yaml`
            default_dict (dict, optional): default dictionary to return \
            if the yaml file does not exist (defaults to :code:`{ }`)

        Returns:
            dict: loaded yaml data
        '''

        res = default_dict

        if FileHandler.__path_exists(path):
            with open(path, 'r') as handler:
                res = yaml.load(handler, Loader=loader)

        return res


