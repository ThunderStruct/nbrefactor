from ...utilities.logger import Logger
from ...search_space.base_search_space import BaseSearchSpace
import abc
from ...performance_metrics.optimizer_metrics import OptimizerMetrics

class BaseOptimizer(abc.ABC):

    def __init__(self, invalid_arch_tolerance, cache_top_n_candidates):
        super().__init__()

        self._current_task = None
        self._best_candidate = None
        self._enable_cl = False

        self._cache_top_n_candidates = cache_top_n_candidates

        self._invalid_arch_tolerance = invalid_arch_tolerance

        self.metrics = OptimizerMetrics()


    def _candidate_scoring(self, models, tasks):
        """
        Calculate scalar scores for all encountered tasks and gets 1 mean score
        per model (weighted average of all tasks based on
        :func:`~BaseTask.importance_weight`)

        Scoring is a relative function that requires more than 1 model

        Returns:
            :obj:`list`: list of scalar scores per model across given tasks. \
            The returned scores' list preserves the order of the passed `models`
        """

        scores_per_model = [0.0 for _ in models]    # init with 0.0 scores
        for task in tasks:
            # each task is scored individually as:
            # a) tasks could have different scoring functions,
            # b) the accuracy obtained on - for instance - ImageNet cannot be
            # compared with that of CIFAR10 (number of classes + complexity) and
            # c) to support multi-modality in the future

            t_scores = task.scoring_func(models, task)
            # [1 scalar score per model]
            for m_idx, score in enumerate(t_scores):
                scores_per_model[m_idx] += score

        # return score mean (across tasks)

        # TODO: rethink the logic here. Do we really need to divide by a
        # constant value when the scores are relative to one another?
        return [score / len(tasks) for score in scores_per_model]


    def candidate_selection(self, models, tasks):
        """
        Default candidate selection across all optimizers (we keep track of the
        best candidate in all cases; storing additional models for population-
        based optimizers can be done on the sub-class level along with their
        memory-management).

        This method uses the internal scoring functions within each given task
        and the training metrics stored on the model level. These scalar scores
        are then used to rank the highest performing model from the given set
        of models and caching it to `self._best_candidate`.

        Args:
            models (:obj:`list`): list of :class:`~Network` objects; all \
            selectable models (newly evaluated model + cached best-performing \
            model in most cases)
            tasks (:obj:`list`): list of all encountered tasks
        """
        assert models and len(models) > 0, 'No models were passed to selection'

        if self._best_candidate is None:
            # intial candidate(s)
            if len(models) == 1:
                self._best_candidate = models[0]
                return
            # else; calculate relative scores for all given models

        # relative scoring between given model(s) and the cached best candidate
        all_candidates = [self._best_candidate]
        all_candidates.extend(models)

        scores = self._candidate_scoring(all_candidates, tasks)
        # sort scores in descending order; highest score at [0]
        # (this is also used for gc later)
        # `scores` preserves ordering; i.e. scores[0] -> models[0]'s score
        sorted_scores = sorted(scores, reverse=True)

        candidate_idx = scores.index(sorted_scores[0])

        if candidate_idx != 0:
            # new best candidate was selected
            old_id = self._best_candidate.id
            old_v = self._best_candidate.version
            new_id = all_candidates[candidate_idx].id
            new_v = all_candidates[candidate_idx].version

            Logger.critical((
                f'A new top candidate has been selected '
                f'(Model ({old_id} v{old_v}) -> Model ({new_id} v{new_v})'
            ))

        self._best_candidate = all_candidates[candidate_idx]

        # housekeeping; delete models outside the caching range
        for s_idx, score in enumerate(sorted_scores):
            if s_idx > self._cache_top_n_candidates - 1:
                # get candidate's index from preserved scores list
                del all_candidates[scores.index(score)]
                # gc triggered once at the end of each NAS cycle for efficiency


    def add_results(self, model_metrics):
        """
        Append aggregated model metrics
        """

        # commit new results to optimizer records
        self.metrics.add_results(model_metrics)


    def assign_task(self, task):
        """
        Sets the current task for the optimizer

        Args:
            task (:class:`~BaseTask`): the task set to be activated
        """

        assert isinstance(task, BaseTask), (
            'Invalid task provided. '
            f'expected type {type(BaseTask)}, received '
            f'{type(task)}'
        )

        assert hasattr(task, 'search_space') and \
        isinstance(task.search_space, BaseSearchSpace), (
            'Invalid search space provided. '
            f'expected type {type(BaseSearchSpace)}, received '
            f'{type(task.search_space)}'
        )

        self._current_task = task


    @property
    def top_metrics(self):
        return self._best_candidate.metrics


    # --------------------------------------------------------------------------
    #   Abstract Methods (implement to conform to this base class)
    # --------------------------------------------------------------------------


    @abc.abstractmethod
    def sample(self):
        """
        Samples a new architecture from the assigned task's search space (
        accessible through `optimizer._current_task.search_space`)

        Returns:
            :obj:`list`: a list of :class:`~Network` (contains $> 1$ model for \
            population-based optimizers, a list of 1 element otherwise ). An \
            empty list or `None` may be returned if the optimizer failed to \
            sample an architecture.
        """

        raise NotImplementedError('Abstract method was not implemented')


