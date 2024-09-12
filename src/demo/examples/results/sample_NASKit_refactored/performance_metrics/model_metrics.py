from ..utilities.functional_utils.misc_utils import flatten_dict
from ..utilities.functional_utils.file_utils import ensure_dir
from ..utilities.config import Config
import os
import pandas as pd

class ModelMetrics:
    """
    The metrics of an individual model (collection of all epochs' results)
    """

    def __init__(self, model_metadata=None):
        self.records = []

        if model_metadata is not None:
            self.model_metadata = flatten_dict({'model': model_metadata.params})


    def set_model_metadata(self, model_metadata):
        """
        Initialize the static model data to be populated across the iteratively-
        added training results
        """
        self.model_metadata = flatten_dict({'model': model_metadata.params})


    def add_eval_metrics(self, eval_metrics):
        """
        Append training metric
        """

        self.records.append(eval_metrics)


    def aggregate(self, on_task_id=None, on_task_version=None, epoch=None):
        """
        Collates the iteratively-added metrics

        Returns:
            :obj:`dict`: the aggregated epochs' metrics
        """
        ret_recs = []
        for record in self.records:
            if on_task_id is not None and on_task_version is not None:
                # aggregate on task ID and version
                if record.task_metadata['task.id'] != on_task_id or \
                record.task_metadata['task.version'] != on_task_version:
                    # skip
                    continue
            elif on_task_id is not None:
                # aggregate on task ID
                if record.task_metadata['task.id'] != on_task_id:
                    # skip
                    continue

            agg_r = record.aggregate()

            collated = [{**self.model_metadata, **sub_rec} \
                        for sub_rec in agg_r]

            ret_recs.extend(collated)

        if epoch:
            if epoch >= len(ret_recs) or epoch < -len(ret_recs):
                # out of range
                return {}
            return ret_recs[epoch]

        return ret_recs


    def save(self, filename, dir='./model_metrics/'):
        """
        Saves model metrics (the accumulated metrics of all models can be
        saved through :class:`~OptimizerMetrics`)
        """

        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        pd.DataFrame(self.aggregate()).to_csv(full_path)


    def __getitem__(self, idx):
        assert isinstance(idx, int) or isinstance(idx, str), (
            'ModelMetrics subscripts must be `int` or `str`'
        )

        if isinstance(idx, str):
            # aggregate model data and records
            agg_recs = self.aggregate()

            vals = []
            for record in agg_recs:
                if idx not in record:
                    raise KeyError(f'ModelMetric key "{idx}" does not exist')
                vals.append(record[idx])
            return vals

        elif isinstance(idx, int):
            return self.aggregate()[idx]


    def __str__(self):
        return str(self.aggregate())


    def __repr__(self):
        return str(self)


