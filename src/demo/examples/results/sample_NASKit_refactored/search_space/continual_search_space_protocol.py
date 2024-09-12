import abc

class CLSearchSpaceProtocol(abc.ABC):
    """
    Methods required to conform to Continual Learning protocol.

    These abstract methods ensure that models can be dynamically extended and
    augmented as needed.
    """


    @abc.abstractmethod
    def random_extend(self, adj_matrix, ops, initial_op_id=None,
                      num_vertices=None, max_attempts=10):
        """
        Topology Extension and Consolidation.

        Similar to :func:`~random_sample()`, this method should sample a random
        extention to a given graph (adjacency matrix + operations list).


        Note: the vertices added could contain identity operations, which
        would essentially mean a connection/edge is added rather than a node.

        Additionally, new edges can probabilistically spawn as we randomly
        generate a new $n \times n$ adjacency matrix and apply logical OR to
        mask the extension with the given subgraph.

        The extended adjacency matrix guarantees the output layer to \
        be the last vertex (`adj_matrix[-1]`) of the DAG.

        Args:
            adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix
            ops (:obj:`list`): the graph's nodes list
            initial_op_id (:obj:`int`, optional): the starting operation ID \
            for this extension (recommended value is the \
            `max(o.id for o in supergraph) + 1`; if `None` is given, you may \
            use :func:`~BaseOperation.get_auto_id()`)
            num_vertices (:obj:`int`): the number of nodes to be extended
            max_attempts (:obj:`int`): maximum attempts to find a valid \
            architecture. If this is exceeded, a \
            `InvalidArchitectureException` with code `3` is raised (caught \
            and handled on the NAS level)

        Returns:
            :obj:`tuple`: a tuple comprising of an adjacency matrix \
            (:class:`np.ndarray`) and a list of :class:`~PartialWrapper`-\
            encapsulated :class:`~BaseOperation` objects (i.e. the layers in \
            the network)
        """

        raise NotImplementedError('Abstract method was not implemented')


