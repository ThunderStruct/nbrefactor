import os
from ..logger import Logger
import shutil


def recursive_dir_delete(path):
    shutil.rmtree(path)


def should_overwrite_path(path, force_overwrite=False):
    if os.path.exists(path):
        if not force_overwrite:
            Logger.warning(f'Path "{path}" is not empty! ',
                            'Set `force_overwrite` to True to overwrite the ',
                            'existing data')
            return False
        else:
            Logger.warning(f'Overwriting path "{path}"...')
            recursive_dir_delete(path)

    return True


def ensure_dir(path, create_if_not_exists):

    if not os.path.exists(path):
        if create_if_not_exists:
            os.makedirs(path)

        return False

    return True

