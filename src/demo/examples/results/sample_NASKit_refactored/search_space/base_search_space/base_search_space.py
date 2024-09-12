from ...utilities.metadata.search_space_metadata import SearchSpaceMetadata
from ..base_operation import BaseOperation
from ...utilities.partial_wrapper import PartialWrapper
import abc
from ...utilities.metadata.operation_metadata import OperationMetadata
from ...utilities.functional_utils.search_space_utils import uniform_operation_weights

class BaseSearchSpace(abc.ABC):

    def __init__(self,
                 num_vertices,
                 operations,
                 encoding,
                 embed_operations_in_metadata=False):
        """
        """
        assert num_vertices > 2, (
            'The given number of vertices is not sufficient'
        )

        self.num_vertices = num_vertices
        self.encoding = encoding
        self.embed_operations_in_metadata = embed_operations_in_metadata

        self._operation_weights_hooks = []

        self.operations = set()

        for op in operations:
            if issubclass(op, BaseOperation):
                # unpack class into partials
                self.operations.update(op.get_all_partials())
            elif isinstance(op, PartialWrapper):
                # already a partial
                self.operations.add(op)
            else:
                raise ValueError(f'Invalid operation type {type(op)}. ' +
                                 'Allowed types are `BaseOperation` ' +
                                 'subclasses or `PartialWrapper` ' +
                                 'encapsulating an `BaseOperation`')

        # default op sampling hook (uniform)
        self.register_sampling_hook(hook_id=0,
                                    hook_func=uniform_operation_weights)

        ## [deprecated] - Normalization occurs during the operation assignment
        ## as the chosen partials will likely be a subset of all partials

        # if len(self.operation_weights) != len(self.operations):
        #     raise ValueError('The provided operation weights are not of ' +
        #                      'the same length as the provided operations')

        # if not math.isclose(sum(self.operation_weights.values()), 1.0):
        #     # float comparison, sum of probs is not 1.0
        #     total = sum(self.operation_weights.values())

        #     # edge case where sum is 0; avoid division by 0
        #     if total != 0:
        #         # normalize weights
        #         self.operation_weights = {key: value / total \
        #                                   for key, value \
        #                                   in self.operation_weights.items()}
        #     else:
        #         # will default to uniform probability
        #         self.operation_weights = None


    def register_sampling_hook(self, hook_id, hook_func):
        """
        Registers a new operation-sampling weight calculation hook. Each hook
        function feeds the next (as a stack; in the order they were registered).

        Hook functions take the following arguments:

            operation_set (:obj:`set`): the given filtered set of operations' \
            partials to calculate weights for
            weights (:obj:`dict`): a dict of weights corresponding to \
            `operation_set`; calculated from previous hooks. The values in \
            this dict are not guaranteed to sum to `1.0`.
            topological_order (:obj:`int`): node's depth in the graph. Could \
            be used to add weight for certain operations in deeper sections \
            of the architecture
            max_depth (:obj:`int`): max depth of the given graph
            prev_ops (:obj:`list`): list of the current node's direct \
            predecessor(s)
            in_shape (:obj:`tuple`): the expected input shape to this node

        Args:
            hook_id (:obj:`hashable`): a hashable ID for the hook (can be used \
            to :func:`~remove_op_sampling_hook`)
            hook_func (:obj:`callable`): the hook function
        """
        assert not any(h[0] == hook_id \
                       for h in self._operation_weights_hooks), (
                           'The given hook ID is already in use'
                           )

        self._operation_weights_hooks.append((hook_id, hook_func))

    def remove_sampling_hook(self, hook_id):
        """
        Removes a previously registered operation-sampling hook function.

        Args:
            hook_id (:obj:`hashable`): a hashable ID for the hook to be removed
        """
        for i, (h_id, hook_func) in enumerate(self._operation_weights_hooks):
            if h_id == hook_id:
                self._operation_weights_hooks.pop(i)
                return

    @property
    def metadata(self):
        ops_metadata = None
        if self.embed_operations_in_metadata:
            # do not embed op_metadata unless explicitly needed as it bloats
            # the results
            ops_metadata = [OperationMetadata.init_from_partial(op) \
                            for op in self.operations]

        return SearchSpaceMetadata(type=type(self).__name__,
                                   num_vertices=self.num_vertices,
                                   encoding=self.encoding,
                                   operations_metadata=ops_metadata)

    def __str__(self):
        return str(self.metadata)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.metadata)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # if self.num_vertices != other.num_vertices:
        #     return False

        # if self.encoding != other.encoding:
        #     return False

        # if len(self.operations) != len(other.operations):
        #     return False

        # if self.operations != other.operations:
        #     # set comparison
        #     return False

        # return True

        # chances of collision are minimal
        return self.metadata == other.metadata

    # --------------------------------------------------------------------------
    #   Abstract Methods (implement to conform to this base class)
    # --------------------------------------------------------------------------

    @abc.abstractmethod
    def random_sample(self, in_shape, out_shape, max_attempts=10):
        """
        Randomly sample an architecture (a DAG + list of uncompiled operations),
        ensuring that:
            1. The graph's adjacency matrix is an upper-triangular matrix (\
            directed and acyclic; i.e. represents a DAG)
            2. The sampled DAG is a rooted-tree \
            (see :func:`~dag_is_rooted_tree()`)
            3. Adjacency matrix has no isolates (0-column/rows)
            4. Has a valid respective list of operations/nodes

        Args:
            in_shape (:obj:`tuple`): the 4-dimensional input shape (BxCxHxW) \
            expected at the input node of the graph
            out_shape (:obj:`tuple`): the 2-dimensional output shape (BxN) \
            expected out of the output node of the graph
            max_attempts (:obj:`int`, optional): maximum number of attempts to \
            sample a valid architecture, after which the function raises \
            `InvalidArchitectureException` with `code=3`. One scenario this \
            would occur is if the majority of architectures in the search \
            space have been evaluated, or the given input/output shapes do not \
            fit with the given operations. This is a stochastic sampling \
            process and so this parameter is used to add tolerance to missed \
            generations. Exceptions raised are caught and handled on the NAS \
            level
            initial_op_id (:obj:`int`, optional): the starting operation ID \
            for this extension (recommended value is the \
            `max(o.id for o in supergraph) + 1`; if `None` is given, you may \
            use :func:`~BaseOperation.get_auto_id()`)

        Returns:
            :obj:`tuple`: a tuple comprising of an adjacency matrix \
            (:class:`np.ndarray`) and a list of :class:`~PartialWrapper`-\
            encapsulated :class:`~BaseOperation` objects (i.e. the layers in \
            the network).
        """
        raise NotImplementedError('Abstract method was not implemented')



class InvalidArchitectureException(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)

        self.code = code


