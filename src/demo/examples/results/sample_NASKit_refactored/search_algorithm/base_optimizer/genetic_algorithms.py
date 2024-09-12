
# class GeneticAlgorithms(BaseOptimizer, CLOptimizerProtocol):

#     def __init__(self, population_size,
#                  invalid_arch_tolerance=10, cache_top_n_candidates=1):

#         super().__init__(invalid_arch_tolerance, cache_top_n_candidates)

#         self.__population = None
#         self.population_size = population_size


#     def __initialize_population(self):
#         """
#         Initializes the population with random models.

#         Returns:
#             List[:class:`~Network`]: A list of randomly initialized models.
#         """
#         new_population = []
#         task = self._current_task
#         in_shape, out_shape = task.shapes

#         for _ in range(self.population_size):
#             adj_matrix, ops = task.search_space.random_sample(
#                 in_shape=in_shape, out_shape=out_shape)

#             model = Network()
#             model.compile(adj_matrix, ops, task.metadata)
#             new_population.append(model)

#         return new_population


#     def __mutate_population(self):
#         """
#         Mutates the existing population to create a new generation.

#         Returns:
#             List[:class:`~Network`]: A list of mutated models.
#         """
#         mutated_population = []
#         task = self._current_task

#         for parent_model in self.__population:
#             adj_matrix, ops = parent_model.get_architecture()

#             try:
#                 new_adj_matrix, new_ops = task.search_space.mutate(adj_matrix, ops)
#                 mutated_model = Network()
#                 mutated_model.compile(new_adj_matrix, new_ops, task.metadata)
#                 if not self.metrics.model_exists(mutated_model):
#                     mutated_population.append(mutated_model)
#             except Exception as e:
#                 print(f"Error during mutation or compilation: {e}")

#         return mutated_population


#     # --------------------------------------------------------------------------
#     # Conforming to Base Optimizer
#     # --------------------------------------------------------------------------


#     def sample(self):
#         """
#         Samples or mutates models based on the current state of the population.
#         If this is the first generation, it initializes the population randomly.
#         Otherwise, it mutates the existing population to create a new generation.

#         Returns:
#             List[:class:`~Network`]: The current generation of models.
#         """

#         if self.__population is None:
#             # Initialize the first generation with random models
#             self.__population = self.__initialize_population()
#         else:
#             # Mutate the existing population to create a new generation
#             self.__population = self.__mutate_population()

#         return self.__population


#     # --------------------------------------------------------------------------
#     # Conforming to Continual Optimizer Protocol
#     # --------------------------------------------------------------------------


#     def fit(self):
#         """
#         Checks if a given task has already been encountered through the model
#         evaluation metrics.

#         This method is responsible for:
#             1- adapting the relevant output layer in a model if the number of
#             classes is altered
#             2- detecting if fine-tuning is required

#         Returns:
#             :obj:`tuple`: a tuple containing the current best performing \
#             model(s) and a boolean indicating whether fine-tuning is required. \
#             It is guaranteed that if fine-tuning is required \
#             (i.e. `ret_tuple[1] == True`) is `True`, then the best performing \
#             model(s) will exist (i.e. `ret_tuple[0]` will not be `None`).
#         """

#         assert self._current_task is not None, (
#             'A task must be set prior to `fit()`. Use '
#             '`optimizer.assign_task()` to set a task.'
#         )

#         task = self._current_task
#         require_fine_tuning = False

#         if self._best_candidate is not None:
#             # Perform class/domain adaptation check
#             for m_idx, m in enumerate(self._best_candidate.tasks_metadata):
#                 if m.id == task.metadata.id:
#                     # class adaptation check
#                     if hasattr(task.metadata, 'classes') and \
#                     not set(task.metadata.classes).issubset(set(m.classes)):
#                         # class-incremental learning
#                         self._best_candidate.adapt_output(task.metadata.id,
#                                                           len(task.classes))

#                         # update task's metadata according to the new increment
#                         # This assumes that
#                         # `SegmentableImageFolder.accumulate_classes` is `True`
#                         new_t = task.metadata
#                         self._best_candidate.tasks_metadata[m_idx] = new_t

#                         require_fine_tuning = True

#                     # domain adaptation check
#                     # if task.boundary.distance_from(m.boundary) > threshold:
#                     #     require_fine_tuning = True
#                     break

#             # _best_candidate could be `None` on the first optimization itr.
#             # if require_fine_tuning is `True`, the model is guaranteed to exist
#             return self._best_candidate, require_fine_tuning

#         return None, False


#     def augment(self, base_model):
#         """
#         """

#         pass


