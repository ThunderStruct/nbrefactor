from ..continual_search_space_protocol import CLSearchSpaceProtocol
from ...utilities.functional_utils.graph_utils import get_topological_orders
from ..base_operation.role_specific_operations import InputStem
import numpy as np
from .base_search_space import BaseSearchSpace
from ...utilities.functional_utils.graph_utils import dag_is_rooted_tree
from ...utilities.functional_utils.graph_utils import trim_isolate_nodes
from ...utilities.functional_utils.graph_utils import predecessor_successor_lists
from ..base_operation import BaseOperation
from ..base_operation.functional_operations import AlignConcat
from ..base_operation.role_specific_operations import OutputStem
from .base_search_space import InvalidArchitectureException


class LWSearchSpace(BaseSearchSpace, CLSearchSpaceProtocol):

    def __init__(self, num_vertices, operations, encoding):
        super(LWSearchSpace, self).__init__(num_vertices, operations,
                                            encoding)

    # def __enforce_architecture_connectivity(self, dag, num_vertices):
    #     # check if a path exists from node 0 to node `num_vertices - 1`
    #     if not nx.has_path(dag, source=0, target=num_vertices - 1):

    #         max_mid_nodes = 1  # maximum intermediate nodes from input to
    #                            # output

    #         for i in range(max_mid_nodes):
    #             mid_node = num_vertices + i
    #             dag.add_node(mid_node)
    #             dag.add_edge(0, mid_node)
    #             dag.add_edge(mid_node, num_vertices - 1)

    #     return dag

    def __generate_topology(self, num_vertices, base_topology=None):
        """

        Args:
            num_vertices (:obj:`int`): the number of vertices in the generated \
            topology
            base_topology (:class:`numpy.ndarray`, optional): an optional \
            base adjacency matrix to be extended. If `None` is given, sample \
            from scratch
        """

        # SINGLE BRANCH TOPOLOGY
        if self.encoding == 'single-branch':
            adj_matrix = np.zeros((num_vertices,
                                   num_vertices), dtype=int)

            for i in range(num_vertices - 1):
                # sequential path from node 0 to node num_vertices-1
                adj_matrix[i, i + 1] = 1

            if base_topology is not None:
                # mask topology excluding the output layer
                # (applying element-wise OR to allow the creation of new edges)
                # Note: only new nodes will be single branched
                base_n = len(base_topology)
                mask = np.logical_or(adj_matrix[:base_n,:base_n], base_topology)
                adj_matrix[:base_n-1,:base_n-1] = mask[:-1,:-1]
                adj_matrix[-1:,-1:] = mask[-1:,-1:] # output layer

            return adj_matrix

        # MULTI-BRANCH TOPOLOGY
        # ----------------------------------------------------------------------
        # Step 1: sample random topology
        adj_matrix = np.random.choice([0, 1],
                                      # non-uniform prob, add: p=[0.7, 0.3])
                                      size=(num_vertices,
                                            num_vertices))

        if base_topology is not None:
            # mask topology excluding the output layer
            # (applying element-wise OR to allow the creation of new edges)
            base_n = len(base_topology)
            mask = np.logical_or(adj_matrix[:base_n,:base_n], base_topology)
            adj_matrix[:base_n-1,:base_n-1] = mask[:-1,:-1]
            adj_matrix[-1:,-1:] = mask[-1:,-1:] # output layer

        # ensure no self-connections / prevent cycles
        adj_matrix = np.triu(adj_matrix, 1)

        # ----------------------------------------------------------------------
        # Step 2: establish intermediary connections (if applicable)
        pred, succ = predecessor_successor_lists(adj_matrix)

        in_idx = 0
        out_idx = len(adj_matrix) - 1

        # ensure all operations have at least 1 input if they have an output
        for op_idx, conns in enumerate(pred):
            if op_idx == in_idx or op_idx == out_idx:
                # skip input/output stems
                continue

            elif len(conns) == 0 and len(succ[op_idx]) > 0:
                # has outputs but no inputs; connect to the input stem
                pred[op_idx].append(in_idx)
                succ[in_idx].append(op_idx)
                adj_matrix[in_idx][op_idx] = adj_matrix[op_idx][in_idx] = 1

            elif len(conns) > 0 and len(succ[op_idx]) == 0:
                # has inputs but no outputs; connect to output stem
                pred[out_idx].append(op_idx)
                succ[op_idx].append(out_idx)
                adj_matrix[out_idx][op_idx] = adj_matrix[op_idx][out_idx] = 1

        # upper triangular matrix (again, post intermediary connections)
        adj_matrix = np.triu(adj_matrix, 1)

        if dag_is_rooted_tree(adj_matrix):
            # valid multi-branch neural network topology
            return adj_matrix

        raise InvalidArchitectureException(code=1,
                                           msg=(
                                               'Could not find a valid '
                                               'topology'
                                          ))


    def __assign_operations(self, adj_matrix,
                            in_stem_shape, out_stem_shape,
                            default_ops=None, initial_op_id=0):
        """
        Takes an architecture as an adjacency matrix + input/output shapes,
        and assigns valid operations for all nodes.

        This method uses DFS to propagate feature shapes across the graph and
        find valid operations/hyperparameters (e.g. too many reduction ops
        could yield a (0x0) output, etc.). Additionally, concatenation is also
        applied where applicable.

        If a `default_ops` parameter is given, the function will extend the
        missing nodes.
        """

        num_vertices = len(adj_matrix)
        pred, succ = predecessor_successor_lists(adj_matrix)

        # set defaults
        ops = [None for _ in range(num_vertices)]
        id_tracker = initial_op_id

        if default_ops is not None:
            # augmenting network
            ops[:len(default_ops)-1] = default_ops[:-1]
        else:
            # initial sample
            ops[0] = InputStem.get_partials(in_stem_shape)[0]
            ops[0].keywords['id'] = id_tracker
            id_tracker += 1

        # compute topological order
        topological_orders = get_topological_orders(pred=pred)

        # find output node idx
        out_idx = num_vertices - 1
        for idx, s_list in enumerate(succ):
            if len(s_list) == 0:
                # output node found; 0 successors
                op_idx = idx
                break

        def dfs_partials(op_idx):
            """Traverse the generated DAG using DFS to infer the operations'
            valid hyperparameters
            """

            nonlocal id_tracker

            if ops[op_idx] is not None:
                # memoized shape
                return ops[op_idx].keywords['out_shape']

            #-------------------------------------------------------------------
            # Compute the input shape to the operation at op_idx.
            # Recursively calculated by propagating back to the input layer

            if len(pred[op_idx]) == 1:
                # single input to op_idx
                in_shape = dfs_partials(pred[op_idx][0])
            else:
                # multi inputs to op_idx
                all_inputs = [dfs_partials(conn) for conn in pred[op_idx]]
                in_shape = AlignConcat.compute_shape(all_inputs)

            #-------------------------------------------------------------------
            # Filter given list of operations for validity based on the in_shape
            # from the previous step

            if op_idx == out_idx:
                # output layer; no need to filter or randomly assign operations
                ops[op_idx] = OutputStem.get_partials(in_shape,
                                                      out_stem_shape[1])[0]
            else:
                # filter operations for validity
                valid_partials = []

                for op in self.operations:
                    valid_partials.extend(op.func.get_partials(in_shape))

                if len(valid_partials) == 0:
                    # will be caught internally on the search space level,
                    # this is the cleanest approach to exit the recursive
                    # control flow
                    exc_msg = 'Could not find a valid operation for one of the'
                    exc_msg += ' graph\'s vertices.'
                    raise InvalidArchitectureException(code=2,
                                                       msg=exc_msg)

                # get weights for valid operations' subset
                # uniform weight of 1.0 as a default
                weights_dict = {op: 1.0 for op in valid_partials}

                all_sampled = [(op, depth) \
                               for op, depth in zip(ops, topological_orders) \
                               if op is not None]

                for hook_id, weight_hook in self._operation_weights_hooks:
                    weights_dict = weight_hook(weights_dict,
                                               topological_orders[op_idx],
                                               max(topological_orders),
                                               [ops[i] for i in pred[op_idx]],
                                               all_sampled,
                                               in_shape)

                    # normalize between hook calls to ensure uniformity
                    weights_sum = sum(weights_dict.values())
                    weights_dict = {key: value / weights_sum \
                                    for key, value in weights_dict.items()}


                # if self._operation_weights_hook is not None and False:
                #     weights_dict = self.\
                #     _operation_weights_hook(valid_partials)
                # else:
                #     # defaults to uniform weights across op. groups
                #     weights_dict = self._uniform_operation_weights(
                #         valid_partials
                #     )

                # # the operation-weights' calculation does not guarantee order
                # ordered_w = []
                # for partial in valid_partials:
                #     ordered_w.append(weights_dict[partial])

                # # normalize weights
                # partial_probs = [w / sum(ordered_w) for w in ordered_w]

                chosen_partial = np.random.choice(list(weights_dict.keys()),
                                                  p=list(weights_dict.values()))

                ops[op_idx] = chosen_partial

            if 'id' not in ops[op_idx].keywords:
                ops[op_idx].keywords['id'] = id_tracker
                id_tracker += 1

            return ops[op_idx].keywords['out_shape']

        dfs_partials(out_idx)  # populate ops list

        return ops


    # --------------------------------------------------------------------------
    # Conforming to Base Search Space
    # --------------------------------------------------------------------------


    def random_sample(self,
                      in_shape, out_shape,
                      max_attempts=10, initial_op_id=0):
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
            the network)
        """
        for _ in range(max_attempts):
            # maximum attempts to find a valid architecture, otherwise raises
            # exception

            try:
                # --------------------------------------------------------------
                # Steps 1 & 2: sample a valid architecture topology
                adj_matrix = self.__generate_topology(self.num_vertices)

                # --------------------------------------------------------------
                # Step 3: trim degree zero nodes (if any exist)
                adj_matrix = trim_isolate_nodes(adj_matrix)

                # --------------------------------------------------------------
                # Step 4: sample random valid operations for the given topology
                ops = self.__assign_operations(adj_matrix,
                                               in_shape, out_shape,
                                               initial_op_id=initial_op_id)


            except InvalidArchitectureException as e:
                if e.code == 1:
                    # failed to establish a valid topology
                    continue
                elif e.code == 2:
                    # could not find a valid operation for one or more nodes
                    continue
                else:
                    raise e
            except Exception as e:
                raise e

            # valid architecture found
            return adj_matrix, ops

        exc_msg = f'Could not find a valid architecture in {max_attempts} '
        exc_msg += 'attempts!'

        raise InvalidArchitectureException(code=3,
                                           msg=exc_msg)


    # --------------------------------------------------------------------------
    # Conforming to the Contiual Search Space Protocol
    # --------------------------------------------------------------------------


    def random_extend(self, adj_matrix, ops, initial_op_id=None,
                      num_vertices=None, max_attempts=10):
        """
        Topology Extension and Consolidation module.

        Similar to :func:`~random_sample()`, this method samples a random
        extention to a given graph (adjacency matrix + operations).


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

        curr_n = len(adj_matrix)
        # default to a quarter of the initial `num_vertices` of the search space
        # with a minimum value of 1 vertex extension
        n_extend = num_vertices if num_vertices else max(self.num_vertices // 4,
                                                         1)
        extended_shape = curr_n + n_extend

        for _ in range(max_attempts):
            # maximum attempts to find a valid architecture, otherwise raises
            # exception

            try:
                # --------------------------------------------------------------
                # Steps 1 & 2: extend adjacency matrix by `n_extend`
                ext_adj = self.__generate_topology(extended_shape,
                                                   base_topology=adj_matrix)

                # --------------------------------------------------------------
                # Step 3: trim degree zero nodes (if any exist)
                ext_adj = trim_isolate_nodes(ext_adj)

                # --------------------------------------------------------------
                # Step 4: sample valid operations for the given topology
                out_shape = ops[-1].keywords['out_shape']
                initial_id = initial_op_id or BaseOperation.get_auto_id()
                ext_ops = self.__assign_operations(ext_adj,
                                                   in_stem_shape=None,
                                                   out_stem_shape=out_shape,
                                                   default_ops=ops,
                                                   initial_op_id=initial_id)
                # replace the output layer's ID (the criteria used for
                # consolidation on the `Network` level)
                ext_ops[-1].keywords['id'] = ops[-1].keywords['id']

            except InvalidArchitectureException as e:
                if e.code == 1:
                    # failed to establish a valid topology
                    continue
                elif e.code == 2:
                    # could not find a valid operation for one or more nodes
                    continue
                else:
                    raise e
            except Exception as e:
                raise e

            # valid architecture found
            return ext_adj, ext_ops

        exc_msg = f'Could not find a valid architecture in {max_attempts} '
        exc_msg += 'attempts!'

        raise InvalidArchitectureException(code=3,
                                           msg=exc_msg)



