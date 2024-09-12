import abc
from ...utilities.metadata.objective_metadata import ObjectiveMetadata

class BaseObjective(abc.ABC):

    def __init__(self):

        assert hasattr(self.__class__, '_DEFAULTS'), (
            '`BaseObjective` subclass must define a static `_DEFAULTS` dict!'
        )

        # objectives' criteria
        self._objs = {}


    def add_criterion(self, metric,
                      polarity=None, score_weight=None,
                      thresholds_enabled=None,
                      min_threshold=None, target_threshold=None):

        cls = self.__class__
        assert metric in cls._DEFAULTS, (
            'An invalid `metric` was provided. Please make sure that the '
            'passed metric exists in the `_DEFAULTS` dict.'
        )

        defaults = cls._DEFAULTS[metric]

        key = defaults['key']

        # do not use `or` for defaulting as some of these values are false-like
        obj_pol = defaults['polarity'] if polarity is None else polarity

        weight = defaults['weight'] if score_weight is None else score_weight

        t_enabled = defaults['thresholds_enabled'] \
        if thresholds_enabled is None else thresholds_enabled

        min_t = defaults['min_threshold'] if min_threshold is None \
        else min_threshold

        target_t = defaults['target_threshold'] if target_threshold is None \
        else target_threshold

        self._objs[metric] = {
            'key': key,
            'polarity': obj_pol,
            'weight': weight,
            'thresholds_enabled': t_enabled,
            'min_threshold': min_t,
            'target_threshold': target_t
        }


    def min_threshold_met(self, eval_metrics):
        """
        Args:
            eval_metrics (:class:`~EvaluationMetrics`): the records' list to \
            assess the threshold over

        Returns:
            :obj:`bool`: whether or not all applicable minimum thresholds were \
            reached. If no threshold is enabled, the function returns `True`
        """

        # check against idx -1; last recorded epoch
        agg_metrics = eval_metrics.aggregate()[-1]

        for criterion in self._objs.values():
            key = criterion['key']

            if not criterion['thresholds_enabled']:
                # skip criterion
                continue

            if key not in agg_metrics:
                # skip criterion
                continue

            if criterion['polarity'] < 0:
                # minimization problem
                if agg_metrics[key] < criterion['min_threshold']:
                    return False
            else:
                # maximization problem
                if agg_metrics[key] > criterion['min_threshold']:
                    return False

        return True


    def target_threshold_met(self, eval_metrics):
        """
        Args:
            eval_metrics (:class:`~EvaluationMetrics`): the records' list to \
            assess the threshold over

        Returns:
            :obj:`bool`: whether or not all applicable target thresholds were \
            reached. If no threshold is enabled, the function returns `False`
        """

        # check against idx -1; last recorded epoch
        agg_metrics = eval_metrics.aggregate()[-1]
        any_assessed = False

        for criterion in self._objs.values():
            key = criterion['key']

            if not criterion['thresholds_enabled']:
                # skip criterion
                continue

            if key not in agg_metrics:
                # skip criterion
                continue

            any_assessed = True   # at least one threshold assessed

            if criterion['polarity'] < 0:
                # minimization problem
                if agg_metrics[key] < criterion['target_threshold']:
                    return False
            else:
                # maximization problem
                if agg_metrics[key] > criterion['target_threshold']:
                    return False

        return any_assessed


    @property
    def score_weights(self):
        return {o['key']: o['weight'] * o['polarity'] \
                for o in self._objs.values()}

    @property
    def metadata_list(self):
        ret_list = [ObjectiveMetadata(o_type=type(self).__name__,
                                      metric_key=o['key'],
                                      polarity=o['polarity'],
                                      score_weight=o['weight'],
                                      thresholds_enabled=\
                                      o['thresholds_enabled'],
                                      min_threshold=o['min_threshold'],
                                      target_threshold=o['target_threshold']) \
                    for o in self._objs.values()]
        return ret_list

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.metadata_list)

    def __eq__(self, other):
        """
        Using __eq__ is much faster than comparing __hash__ values
        (dependent on __str__ values of datasource + search_space which
        potentially encapsulates a lot of operations)
        """
        if not isinstance(other, self.__class__):
            return False

        return self.metadata_list == other.metadata_list

    def __hash__(self):
        return hash(self.metadata_list)


