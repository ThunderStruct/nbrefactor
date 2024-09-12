import torch
import abc
import numpy as np

class BaseXAIInterpreter(abc.ABC):

    def __init__(self):
        super().__init__()


    def set_caching(self, false_pred_count, true_pred_count):

        self.false_preds = false_pred_count
        self.true_preds = true_pred_count

        self.cached_false, self.cached_true = [], []


    @abc.abstractmethod
    def register_hooks(self, model):
        """
        Register the layers' backward hooks.

        Do not store gradients or memory-intensive attributes
        during this phase, as the hooks are applied prior to training
        (i.e. could result in storing all training gradients)

        For a single pass/batch analysis, register the hooks during the
        interpretation phase.
        """
        raise NotImplementedError('Abstract method was not implemented')


    @abc.abstractmethod
    def interpret(self, model):
        pass


    def cache_predictions(self, inputs, preds):
        """
        """
        assert hasattr(self, 'false_preds') and hasattr(self, 'true_preds'), (
            '`set_caching()` must be called prior to caching predictions!'
        )

        false_count = self.false_preds
        true_count = self.true_preds

        # cache for interpretation
        if false_count > 0 and len(self.cached_false) < false_count:
            remaining = false_count - len(self.cached_false)
            if remaining < 1:
                return
            # filter indices
            all_false_preds = np.where(preds.cpu().numpy() == False)[0]
            valid_indices = all_false_preds[:min(remaining,
                                                 len(all_false_preds) - 1)]
            self.cached_false.extend(inputs[valid_indices].cpu())

        if true_count > 0 and len(self.cached_true) < true_count:
            remaining = true_count - len(self.cached_true)
            if remaining < 1:
                return
            # filter indices
            all_true_preds = np.where(preds.cpu().numpy() == True)[0]
            valid_indices = all_true_preds[:min(remaining,
                                                len(all_true_preds) - 1)]
            self.cached_true.extend(inputs[valid_indices].cpu())


    def get_cached_predictions(self, shuffle=True):
        # stack cached predictions along the first dimension
        preds = torch.stack(self.cached_false + self.cached_true, dim=0)

        if shuffle:
            preds = preds[torch.randperm(preds.size(0))]

        return preds

    def reset(self):
        self.cached_false, self.cached_true = [], []


    def normalize_scores(self, scores):
        """
        Normalizing the scores to prevent exploding gradients/scores
        Regularize by 1e-10 to prevent division by 0 or exploding values.

        Modifies the scores dict in-place.

        Args:
            scores (:obj:`dict`): the scores dict, wherein the keys are module \
            IDs, and the values are the scores
        """

        values = list(scores.values())

        min_score = min(values)
        max_score = max(values)

        for key in scores:
            scores[key] = (scores[key] - min_score) / \
                          (max_score - min_score + 1e-10)


    @property
    def name(self):
        cls = self.__class__
        assert hasattr(cls, 'NAME'), (
            'XAI interpreters must have a static `NAME` attribute'
        )

        return cls.NAME


