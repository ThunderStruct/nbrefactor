import abc

class CLOptimizerProtocol(abc.ABC):
    """
    Methods required to conform to Continual Learning protocol.

    These abstract methods ensure that the optimizer can handle all three types
    of incremental learning (domain-/class-/task-incremental).
    """

    @abc.abstractmethod
    def fit(self):
        """
        Checks if a given task has already been encountered through the model
        evaluation metrics.

        This method is responsible for:
            1- adapting the relevant output layer in a model if the number of
            classes is altered
            2- detecting if fine-tuning is required

        Returns:
            :obj:`tuple`: a tuple containing the current best performing \
            model(s) and a boolean indicating whether fine-tuning is required. \
            It is guaranteed that if fine-tuning is required \
            (i.e. `ret_tuple[1] == True`) is `True`, then the best performing \
            model(s) will exist (i.e. `ret_tuple[0]` will not be `None`).
        """

        raise NotImplementedError('Abstract method was not implemented')


    @abc.abstractmethod
    def augment(self, base_model):
        """
        Identifies the type of incremental learning and augments the model
        accordingly. This method is similar to :func:`~BaseOptimizer.sample()`
        with the exception that it takes a existing `base_model` and either:

            1- Extends the model; increases the complexity of a task's \
            sub-graph within the model (by adding vertices). This is to \
            overcome growing complexities in Domain-/Class-Incremental \
            problems, or
            2- Samples a new architecture for a Task-Incremental scenario and \
            consolidates it with the existing `base_model` (should incentivize \
            operation overlapping).

        This is used when a model's performance drops due to a shift in the
        underlying distribution of the dataset. In most scenarios, fine-tuning
        the model overcomes the shift, however, when the dataset's complexity
        increases, this method might be called to adapt the respective portion
        of the model accordingly.

        Args:
            base_model (:class:`~Network`): the base model comprising of \
            multiple sub-graphs for all encountered tasks.
        Returns:
            :obj:`list`: a list of :class:`~Network` objects, containing the \
            augmented model(s)
        """

        raise NotImplementedError('Abstract method was not implemented')


