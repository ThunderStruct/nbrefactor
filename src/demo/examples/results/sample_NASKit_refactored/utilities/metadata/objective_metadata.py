from .metadata import Metadata

class ObjectiveMetadata(Metadata):
    def __init__(self, o_type, metric_key, polarity, score_weight,
                 thresholds_enabled, min_threshold, target_threshold, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'type': o_type,
            'metric_key': metric_key,
            'polarity': polarity,
            'score_weight': score_weight,
            'thresholds_enabled': thresholds_enabled,
            'min_threshold': min_threshold,
            'target_threshold': target_threshold
        }

        super(ObjectiveMetadata, self).__init__(params=params, **kwargs)


