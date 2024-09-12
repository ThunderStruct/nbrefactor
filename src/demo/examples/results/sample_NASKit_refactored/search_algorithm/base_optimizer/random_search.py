from ...utilities.functional_utils.search_space_utils import incentivize_operation_overlap_hook_factory
from .base_optimizer import BaseOptimizer
from ..continual_optimization_protocol import CLOptimizerProtocol
from ...utilities.functional_utils.graph_utils import get_topological_orders
from ...search_space.base_search_space import InvalidArchitectureException
from ...utilities.logger import Logger
from ...search_space.model.network import Network

class RandomSearch(BaseOptimizer, CLOptimizerProtocol):

    def __init__(self, invalid_arch_tolerance=10, cache_top_n_candidates=1):
        super().__init__(invalid_arch_tolerance, cache_top_n_candidates)


    # --------------------------------------------------------------------------
    # Conforming to Base Optimizer
    # --------------------------------------------------------------------------


    def sample(self):
        """
        Samples a new architecture from the assigned task's search space.

        Returns:
            :obj:`list`: a list of :class:`~Network` objects, containing the \
            sampled model
        """
        assert self._current_task is not None, (
            'A task must be set prior to sampling'
        )

        task = self._current_task
        in_shape, out_shape = task.shapes

        for tolerance_idx in range(self._invalid_arch_tolerance):
            # two types of invalid architectures: previously evaluated,
            # or dysfunctional network (couldn't find a valid layer for
            # one or more nodes, etc.)
            try:
                # sample a random architecture from the task's search space
                adj_matrix, ops = task.search_space.random_sample(
                    in_shape=in_shape,
                    out_shape=out_shape,
                    max_attempts=self._invalid_arch_tolerance - tolerance_idx
                )

                model = Network()
                model.compile(adj_matrix, ops, task.metadata)

                if self.metrics.model_exists(model):
                    # model isomorphism has been evaluated
                    del model
                    continue

                # return as a list to conform with population-based optimizers
                return [model]

            except InvalidArchitectureException as e:
                if e.code == 3:
                    # failed to find a valid architecture
                    return None
                raise e
            except Exception as e:
                raise e


    # --------------------------------------------------------------------------
    # Conforming to Continual Optimizer Protocol
    # --------------------------------------------------------------------------


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

        assert self._current_task is not None, (
            'A task must be set prior to `fit()`. Use '
            '`optimizer.assign_task()` to set a task.'
        )

        task = self._current_task
        require_fine_tuning = False

        if self._best_candidate is not None:
            # Perform class/domain adaptation check
            for m_idx, m in enumerate(self._best_candidate.tasks_metadata):
                if m.id == task.metadata.id:
                    # class adaptation check
                    if hasattr(task.metadata, 'classes') and \
                    not set(task.metadata.classes).issubset(set(m.classes)):
                        # class-incremental learning
                        self._best_candidate.adapt_output(task.metadata.id,
                                                          len(task.classes))

                        # update task's metadata according to the new increment
                        # This assumes that
                        # `SegmentableImageFolder.accumulate_classes` is `True`
                        new_t = task.metadata
                        self._best_candidate.tasks_metadata[m_idx] = new_t

                        require_fine_tuning = True

                    # domain adaptation check
                    # if task.boundary.distance_from(m.boundary) > threshold:
                    #     require_fine_tuning = True
                    break

            # _best_candidate could be `None` on the first optimization itr.
            # if require_fine_tuning is `True`, the model is guaranteed to exist
            return self._best_candidate, require_fine_tuning

        return None, False


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
            consolidates it with the existing `base_model` (incentivizes \
            operation overlapping).

        This is used when a model's performance drops due to a shift in the
        underlying distribution of the dataset. In most scenarios, fine-tuning
        the model overcomes the shift, however, when the dataset's complexity
        increases, this method might be called to adapt the respective portion
        of the model accordingly.

        Args:
            base_model (:class:`~Network`): the base model comprising of \
            multiple sub-graphs for all given tasks.
        Returns:
            :obj:`list`: a list of :class:`~Network` objects, containing the \
            augmented model
        """
        assert self._current_task is not None, (
            'A task must be set prior to sampling'
        )

        assert base_model is None or isinstance(base_model, Network), (
            'The provided model must be of type `Network`'
        )

        task = self._current_task
        in_shape, out_shape = task.shapes
        initial_id = max([o.id for o in base_model.nodes()]) + 1

        for tolerance_idx in range(self._invalid_arch_tolerance):
            # two types of invalid architectures: previously evaluated,
            # or non-functional DNN
            try:
                adj_matrix = None
                ops = None

                if any([True for t in base_model.tasks_metadata \
                        if t.id == task.id]):
                    # task was previously encountered; DIL or CIL -> extend

                    # extend existing subgraph to increase its complexity
                    Logger.critical((
                        f'Extending subgraph for task ({task.id} '
                        f'v.{task.version})...'
                    ))

                    subgraph = base_model.get_subgraph(task.id)
                    decomp_ops = [op.decompile() for op in subgraph[1]]

                    adj_matrix,\
                    ops = task.search_space.random_extend(subgraph[0],
                                                          decomp_ops,
                                                          initial_op_id=\
                                                          initial_id)

                else:
                    # new task; TIL

                    # enable overlapping incentivization
                    depths = get_topological_orders(base_model.adj_matrix)
                    zipped_ops = zip(base_model.nodes(), depths)
                    decomp_ops = [(node.decompile(), depth) \
                                  for node, depth in zipped_ops]

                    hook_id = '__op_overlap_hook'
                    hk = incentivize_operation_overlap_hook_factory(decomp_ops)
                    task.search_space.register_sampling_hook(hook_id,
                                                             hk)

                    # sample new architceture that has incentivized
                    # intersection with `base_model`
                    adj_matrix, ops = task.search_space.random_sample(
                        in_shape=in_shape,
                        out_shape=out_shape,
                        max_attempts=self._invalid_arch_tolerance - \
                        tolerance_idx,
                        initial_op_id=initial_id)

                    task.search_space.remove_sampling_hook(hook_id)

                # clone model in case we need to revert back to older model
                model = base_model.clone()
                model.compile(adj_matrix, ops, task.metadata)

                if self.metrics.model_exists(model):
                    # model isomorphism has been evaluated
                    del model
                    continue

                # return as a list to conform with population-based optimizers
                return [model]

            except InvalidArchitectureException as e:
                if e.code == 3:
                    # failed to find a valid architecture
                    return None
                raise e
            except Exception as e:
                raise e


