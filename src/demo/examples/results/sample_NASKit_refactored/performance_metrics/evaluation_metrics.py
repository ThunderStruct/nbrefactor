from ..utilities.functional_utils.misc_utils import flatten_dict
import time

class EvaluationMetrics:
    """
    Candidate evaluation metrics (1 record per epoch; batch-level metrics
    could be saved in logs in an unstructured/plain-text format)
    """

    def __init__(self, task_metadata):

        self.records = []
        self.task_metadata = flatten_dict({'task': task_metadata.params})


    def add_metrics(self, metrics, epoch, start_time):
        """
        Record an epoch's metrics
        """

        start_time_fmt = time.strftime('%Y/%m/%d, %H:%M:%S',
                                       time.localtime(start_time))

        rec = {
            'epoch': epoch,
            'start_time': start_time_fmt,
            'duration': time.time() - start_time
        }
        rec = {**metrics, **rec}

        self.records.append(rec)


    def aggregate(self):
        """
        Collates the iteratively-added metrics

        Returns:
            :obj:`dict`: the aggregated epochs' metrics
        """

        return [{**self.task_metadata, **record} for record in self.records]


    def __getitem__(self, idx):
        assert isinstance(idx, str), (
            'Non-String indices are not supported for ModelMetrics objects'
        )

        # aggregate model data and records
        agg_recs = self.aggregate()

        vals = []
        for record in agg_recs:
            if idx not in record:
                raise KeyError(f'metric key "{idx}" does not exist')
            vals.append(record[idx])
        return vals


    def __str__(self):
        return str([{**self.task_metadata, **record} \
                    for record in self.records])


    def __repr__(self):
        return str(self)


