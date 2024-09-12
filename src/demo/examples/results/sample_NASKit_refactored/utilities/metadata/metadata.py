from ..functional_utils.file_utils import ensure_dir
from ..logger import Logger
from copy import deepcopy
from ..config import Config
import json
import os

class Metadata:

    def __init__(self, params=None, **kwargs):
        self.params = dict(params if params else {}, **kwargs)

        for key, value in self.params.items():
            setattr(self, key, value)

    def pretty_print(self):
        dump_dict = deepcopy(self.params)
        for key, val in dump_dict.items():
            dump_dict[key] = str(val)

        return json.dumps(dump_dict, indent=4, sort_keys=True)

    def save(self, dir, filename):

        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        dump_dict = deepcopy(self.params)
        for key, val in dump_dict.items():
            dump_dict[key] = str(val)
        try:
            with open(full_path, 'w') as f:
                f.write(json.dumps(dump_dict))
        except Exception as e:
            Logger.debug(f'Error writing metadata log to {full_path}: {e}')

    def __str__(self):
        return ','.join([f'{k}={str(v)}' for k, v in self.params.items()])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.params == other.params

    def __hash__(self):
        return hash(str(self))


