from ..utilities.config import Config
import os
from ..utilities.functional_utils.file_utils import ensure_dir
import pandas as pd


class OptimizerMetrics:
    """
    The performance metrics of all evaluated models (collection of
    :class:`~ModelMetrics` objects)
    """

    def __init__(self):
        """
        """

        self.records = pd.DataFrame()


    def add_results(self, model_metadata):
        """
        """
        self.records = pd.concat([self.records, pd.DataFrame(model_metadata)],
                                 ignore_index=True)


    def model_exists(self, model):
        if 'model_hash' not in self.records:
            return False

        return hash(model) in self.records['model_hash'].values


    def save(self, filename, dir='./nas_results/'):
        """
        """

        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        self.records.to_csv(full_path)


    def __str__(self):
        return str(self.records)


    def __repr__(self):
        return str(self)


