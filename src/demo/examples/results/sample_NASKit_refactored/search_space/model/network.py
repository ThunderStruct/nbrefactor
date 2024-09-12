import json
from ..base_operation.functional_operations import ShapeMatchTransform
from ...utilities.logger import Logger
from collections import defaultdict
import torch
from ...performance_metrics.model_metrics import ModelMetrics
import torch.nn as nn
from ..base_operation.role_specific_operations import OutputStem
import networkx as nx
from ...utilities.partial_wrapper import PartialWrapper
from ...visualization import visualize_network
from copy import deepcopy
import numpy as np
from ...utilities.metadata.model_metadata import ModelMetadata
from ..base_operation.functional_operations import AlignConcat
from ...utilities.metadata import Metadata


class Network(nn.Module, nx.DiGraph):
    """
    The DAG representation of a neural architecture (or a motif/cell
    subgraph).
    """

    __SERIAL_ID = 0

    def __init__(self, adj_matrix=None, operations=None, task_metadata=None):
        # explicit superclass init to ensure expected MRO
        nx.DiGraph.__init__(self)
        nn.Module.__init__(self)

        self.id = Network.__SERIAL_ID
        self.version = 0        # gets incremented with each graph update
        Network.__SERIAL_ID += 1

        self.task_map = {}      # task_id : model heuristics
                                #           (adj_matrix, ops, predecessors)
                                # we memoize preds to off-load the recursive
                                # training complexity

        self.tasks_metadata = []
        self.metrics = ModelMetrics()

        if adj_matrix and operations and task_metadata:
            # compile on init
            self.compile(adj_matrix, operations, task_metadata)


    #---------------------------------------------------------------------------
    # GRAPH COMPILATION

    def __update_graph(self, adj_matrix, ops):
        """
        Initialize or alter the Network from the given graph (adjacency matrix
        and operations).

        This method 1) adds nodes from the given `ops` list if they do not
        already exist in the supergraph (must match `op.id` to qualify).
        2) Similarly, we add edges from the given `adj_matrix` if they do not
        exist. 3) Remove isolates in case edge-removal resulted in layer
        isolation. 4) We compute the predecessors for this particular subgraph
        (the topology may only be altered in this function, thus the
        predecessors cannot change otherwise).

        Args:
            adj_matrix (:class:`np.ndarray`): adjacency matrix-encoded DAG
            ops (:obj:`list`): list of compiled ops corresponding the \
            adj_matrix topology
        """

        # # add nodes to graph with additional node attributes (index/depth)
        # depths = get_nodes_depths(adj_matrix)
        # new_nodes_idx = len(self.operations) - len(added_nodes)
        # nodes = [(op, {'index': idx, 'depth': depths[idx]}) \
        #          for idx, op in enumerate(self.operations) \
        #          if idx >= new_nodes_idx and op is not None]

        # ----------------------------------------------------------------------
        # Filter new nodes to add to the graph
        nodes_to_add = []

        for node in ops:
            node_exists = False
            for existing_node in self.nodes():
                if node.id == existing_node.id:
                    # node already exists, do not add
                    node_exists = True
                    break

            if not node_exists:
                # new node, add it
                nodes_to_add.append(node)

        self.add_nodes_from([(node, {'signature': node.metadata.signature}) \
                             for node in nodes_to_add])

        # ----------------------------------------------------------------------
        # Modify graph edges

        subgraph_nodes = []
        preds = defaultdict(list)
        for i in range(len(adj_matrix)):
            op_i = self.get_node_by_id(ops[i].id)
            subgraph_nodes.append(op_i)
            for j in range(len(adj_matrix)):
                op_j = self.get_node_by_id(ops[j].id)

                if adj_matrix[i][j] == 1:
                    preds[op_j.id].append(op_i)
                    # edge should exist (add if not exists)
                    if not self.has_edge(op_i, op_j):
                        self.add_edge(op_i, op_j)

                else:
                    # edge should not exist (remove if exists)
                    if self.has_edge(op_i, op_j):
                        self.remove_edge(op_i, op_j)

        # ----------------------------------------------------------------------
        # Check for isolates (0 degree nodes) resulting from edge removal,
        # remove from graph accordingly
        self.remove_nodes_from(list(nx.isolates(self)))

        return (adj_matrix, subgraph_nodes, preds)

    # def __out_channels_dfs(self, op_idx, input_conns):
    #     """
    #     Recursively traverse the graph from `op_idx` to the root node, \
    #     populating the `out_channels` argument for the operations

    #     Args:
    #         op_idx (int): the operation's index to traverse back to the \
    #         input from
    #         input_conns (list): 2D array of the operations' input connections
    #     """

    #     if 'out_channels' in self.operations[op_idx].keywords:
    #         return self.operations[op_idx].keywords['out_channels']

    #     tot_channels = 0
    #     for in_conn in inputrecon_conns[op_idx]:
    #         channels = self.__out_channels_dfs(in_conn, input_conns)
    #         if 'out_channels' not in self.operations[in_conn].keywords:
    #             self.operations[in_conn].keywords['out_channels'] = channels
    #         tot_channels += channels

    #     return tot_channels


    # def __out_shape_dfs(self, op_idx):
    #     """
    #     Recursively traverse the graph from `op_idx` to the root node, \
    #     calculating the output shape of each operation

    #     Args:
    #         op_idx (int): the operation's index to traverse back to the \
    #         input from
    #     """
    #     assert self.output_shapes[0] is not None, (
    #           'Base case missing for the out_shape DFS'
    #     )
    #     if self.output_shapes[op_idx] is not None:
    #         # memoization
    #         return self.output_shapes[op_idx]

    #     if len(self.input_conns[op_idx]) == 1:
    #         prev_out = self.__out_shape_dfs(self.input_conns[op_idx][0])
    #         self.output_shapes[op_idx] = self.compiled_ops[op_idx]\
    #                                          .calculate_feature_map(prev_out)
    #         return self.output_shapes[op_idx]
    #     else:
    #         max_in_shape = (0, 0)
    #         # multi-input op
    #         for in_conn in self.input_conns[op_idx]:
    #             prev_out = self.__out_shape_dfs(in_conn)
    #             max_in_shape = (max(max_in_shape[0], prev_out[0]),
    #                                 max(max_in_shape[1], prev_out[1]))

    #         self.output_shapes[op_idx] = max_in_shape
    #         return self.output_shapes[op_idx]


    def compile(self, adj_matrix, operations, task_metadata):
        """
        Compile the given (sub-)graph (list of uninstantiated operations and
        adjacency matrix) and add the edges of the :class:`nx.DiGraph`
        accordingly.

        Args:
            adj_matrix (:class:`numpy.ndarray`): adjacency matrix-encoded DAG
            operations (:obj:`list`): ordered list of (uninstantiated)
            operations corresponding the edges of the adjacency matrix, each \
            in the form of a partial function (:class:`~PartialWrapper`)
            task_metadata (:class:`~TaskMetadata`): the task data associated \
            with the given model; used to keep track of tasks and their
            sub-graphs within extended models
        """

        # list of compiled operations for this sub-graph
        compiled_ops = []

        for op in operations:
            if isinstance(op, PartialWrapper):
                # compile the operation temporarily to equate with existing
                # operations in the super-graph.

                # By design, if the op shares the same unique ID with an
                # existing node, we are intended to reuse it.
                temp_compiled = op()
                op_is_precompiled = False

                for precompiled_op in self.nodes():
                    if precompiled_op.id == temp_compiled.id:
                        if isinstance(precompiled_op, OutputStem):
                            # reshape and preserve weights if applicable
                            in_shape = temp_compiled.in_shape
                            num_classes = temp_compiled.num_classes
                            precompiled_op.reshape(in_shape=in_shape,
                                                   num_classes=num_classes)

                        # operation already compiled, do not recompile
                        compiled_ops.append(precompiled_op)
                        op_is_precompiled = True
                        break
                if not op_is_precompiled:
                    compiled_ops.append(temp_compiled)
            else:
                # [DEPRECATED]
                # ~~# isolate node, add `None` to preserve index order
                # self.operations.append(None)~~

                # [revised search space requirements]: `Network` does not accept
                # isolate/`None` nodes. Only `PartialWrapper`s of
                # `BaseOperation`s are accepted.

                raise ValueError(
                    f'An invalid operation of type `{type(op)}` was given. '
                    'Only `PartialWrapper`s of `BaseOperation`s are '
                    'accepted'
                )

        # if label_nodes:
        #     label_map = {idx: f'{idx}:{op.op_name}' \
        #     for idx, op in enumerate(self.compiled_ops) if op is not None}
        #     nx.relabel_nodes(self, label_map, copy=False)

        # prev_subgraph = None
        # if task_metadata.id in self.task_map:
        #     prev_subgraph = self.get_subgraph(task_metadata.id)

        subgraph = self.__update_graph(adj_matrix, compiled_ops)

        self.task_map[task_metadata.id] = subgraph
        self.tasks_metadata.append(task_metadata)

        self.metrics.set_model_metadata(self.metadata)
        self.version += 1


    #---------------------------------------------------------------------------
    # CONTINUAL LEARNING METHODS

    def adapt_output(self, task_id, new_class_count):
        """
        Adapts the model for a new output space.
        The `task_id` is used to identify the output layer to be modified.

        Note: This method does not fine-tune or retrain the model, just merely
        resizes the output layer for the given task's (sub-)graph *whilst*
        masking or preserving the existing trained weights in said layer.

        Args:
            task_id (:obj:`int`): the task ID associated with the given model; \
            used to keep track of tasks and their sub-graphs within extended
            models
            new_class_count (:obj:`int`): the new number of output features \
            for the given `task_id`
        """

        out_shape_pre = self.get_output_layer(task_id).out_shape
        self.get_output_layer(task_id).reshape(num_classes=new_class_count)
        out_shape_post = self.get_output_layer(task_id).out_shape

        Logger.critical(f'Model ID ({self.id}) adapted its output ' + \
                    f'from {out_shape_pre} to {out_shape_post}...')

    # def extend_network(self, task_id, adj_matrix, operations):
    #     """
    #     Compile the model given the list of uninstantiated operations and the
    #     adjacency matrix. This function instantiates the operations and adds
    #     them on the edges of the :class:`nx.DiGraph`
    #     Args:
    #         task_id (:obj:`int`): the task ID associated with the given model\
    #         ; used to keep track of tasks and their sub-graphs within extended
    #         models
    #         adj_matrix (:class:`numpy.ndarray`): adjacency matrix-encoded DAG
    #         operations (:obj:`list`): ordered list of (uninstantiated)
    #         operations corresponding the edges of the adjacency matrix, each \
    #         in the form of a partial function
    #     """

    #     if not hasattr(self, 'operations'):
    #         # no existing graph, compile
    #         return self.compile_network(adj_matrix, operations, task_id)

    #     # extend model
    #     self.metrics = ModelMetrics(self.metadata)


    #---------------------------------------------------------------------------
    # TORCH METHODS

    def forward(self, x):
        """
        Perform a forward pass through the network graph.\
        This process is performed using DFS since the connections' order is \
        unknown a priori

        Args:
            x (:class:`torch.Tensor`): the network's input data

        Returns:
            :class:`torch.Tensor`: the network's output data
        """

        assert hasattr(self, 'active_task') \
        and self.active_task is not None, (
            'A task needs to be activated prior to model training'
        )

        return self.__recursive_forward(x)

    def __recursive_forward(self, x):
        """
        Recursively executes a forward pass for the active task's subgraph.

        Args:
            x (:class:`torch.Tensor`): the network's input data

        Returns:
            :class:`torch.Tensor`: the network's output data
        """

        # intermediate tensor memoization
        outputs = {}

        # get computed predecessors for the active task
        _, _, all_preds = self.task_map[self.active_task]

        def dfs_outputs(op):
            # memoization check; already processed
            if op.id in outputs:
                return outputs[op.id]

            preds = all_preds[op.id]

            # root base case (input node / no predecessors)
            if len(preds) == 0:
                outputs[op.id] = op(x)
                return outputs[op.id]

            # single input operations
            if len(preds) == 1:
                outputs[op.id] = op(dfs_outputs(preds[0]))
                return outputs[op.id]

            # multi-input operations
            input_tensors = []
            for pred_op in preds:
                if pred_op.id not in outputs:
                    outputs[pred_op.id] = dfs_outputs(pred_op)

                input_tensors.append(outputs[pred_op.id])

            aggregate = AlignConcat.functional(input_tensors)

            # TODO: I don't like shape transforms. Constrain the
            # topology-extension properly to ensure shape-match and remove
            # this. The whole point of this framework is to get rid of paddings
            # and up-/down-sampling to reduce overhead. I hate this.
            aggregate = ShapeMatchTransform.functional(input=aggregate,
                                                       shape=op.in_shape)
            outputs[op.id] = op(aggregate)
            return outputs[op.id]

        # DFS starting at the output node
        out = dfs_outputs(self.get_output_layer(task_id=self.active_task))

        # clear intermediate feature maps
        del outputs

        return out

    def parameters(self, recurse=True):
        """
        Override :func:`nn.Module.parameters()` function.

        PyTorch is normally expected to walk all members of this class
        recursively, and accumulate parameters from:
            1- anything manually registered with PyTorch's \
            :func:`register_parameter()`
            2- any member sub-classing :class:`nn.Module` \
            3- any element in a `nn.ModuleList` (essentially same as rule 2)

        Since we are maintaing the modules as the nodes of :class:`nx.DiGraph`,
        we could either maintain a parallel list of modules to match the
        graph's nodes, or override this method to yield the parameters in the
        network dynamically. I thought the former is too cumbersome and
        vulnerable to mistakes, so we dynamically yield the parameters from our
        graph's nodes instead.

        Args:
            recurse (:obj:`bool`): whether we should recursively walk the \
            `nn.Module`

        Returns:
            :class:`nn.Parameter`: the yielded parameter (in the form of a \
            generator)
        """

        for node in self.nodes():
            if isinstance(node, nn.Module):
                for param in node.parameters(recurse=recurse):
                    yield param

    def cuda(self):
        """
        Override :func:`nn.Module.cuda()` function.

        Similar to the :func:`nn.Module.parameters()` function, the lack of
        module registration (into a class member :class:`nn.Sequential` or
        :class:`nn.ModuleList`) raises an issue when setting `cuda` or `cpu`
        settings.

        We simply traverse our DAG and apply the setting manually.

        """

        for node in self.nodes():
            if isinstance(node, nn.Module):
                node.cuda()

    def cpu(self):
        """
        Override :func:`nn.Module.cpu()` function.

        Similar to the :func:`nn.Module.parameters()` function, the lack of
        module registration (into a class member :class:`nn.Sequential` or
        :class:`nn.ModuleList`) raises an issue when setting `cuda` or `cpu`
        settings.

        We simply traverse our DAG and apply the setting manually.

        """

        for node in self.nodes():
            if isinstance(node, nn.Module):
                node.cpu()

    #---------------------------------------------------------------------------
    # PROPERTIES

    @property
    def adj_matrix(self):
        """
        Gets the network's adjacency matrix (including all sub-graphs).

        Returns:
            :class:`numpy.ndarray`: adjacency matrix representation of the \
            :class:`~Network`'s DAG
        """
        if len(self.nodes) > 0:
            return nx.adjacency_matrix(self).toarray()
        else:
            return np.ndarray([])

    @property
    def flops(self):
        """
        Calculates the total MFLOPs of all the operations in the graph.

        Returns:
            :obj:`float`: the model's MFLOPs per forward pass
        """

        return sum([op.flops for op in self.nodes()])

    @property
    def total_params(self):
        """
        Calculates the total number of parameters in the model

        Returns:
            :obj:`int`: the total count of parameters in the model
        """

        return sum(p.numel() for p in self.parameters())

    @property
    def learnable_params(self):
        """
        Calculates the total number of learnable parameters in the model

        Returns:
            :obj:`int`: the total count of learnable parameters in the model
        """

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def wl_hash(self):
        """
        Get Weisfeiler-Lehman Hash representing the graph.
        This is used to avoid duplicitous evaluations of isomorphic graphs,
        as well as to structure the results.

        According to NetworkX docs:

            If no node or edge attributes are provided, the degree of each node
            is used as its initial label. Otherwise, node and/or edge labels
            are used to compute the hash.

        We specify the degree of the nodes based on each operation's
        structural (non-unique) signature. In other words, two
        :class:`~BatchNormalization` operations should be considered identical
        nodes (same `signature` value), but their hashes will be different due
        to their unique `id` attributes.

        Returns:
            :obj:`str`: the graph's WL hash
        """

        return nx.weisfeiler_lehman_graph_hash(self, node_attr='signature')

    @property
    def metadata(self):
        """
        Encapsulates and returns the object's metadata for logging, hashing,
        and equating purposes

        Returns:
            :class:`~ModelMetadata`: the object's metadata
        """
        nodes = [str(op).strip() for op in self.nodes(data=True)]
        adj_matrix = self.adj_matrix.tolist()

        task_map = {}
        for k, subgraph in self.task_map.items():
            preds = {key: [o.id for o in ol] \
                     for key, ol in self.task_map[k][2].items()}
            task_map[k] = (subgraph[0].tolist(),
                           [o.id for o in subgraph[1]],
                           preds)

        return ModelMetadata(id=self.id,
                             version=self.version,
                             wl_hash=self.wl_hash,
                             model_hash=hash(self),
                             serialized_graph=self.serialize(),
                             mflops=self.flops / 1e6,
                             total_params=self.total_params,
                             learnable_params=self.learnable_params,
                             adj_matrix=adj_matrix,
                             nodes=nodes,
                             task_map=str(task_map),
                             tasks_metadata=self.tasks_metadata)


    #---------------------------------------------------------------------------
    # SERIALIZATION METHODS

    @staticmethod
    def deserialize(graph_str):
        """
        """

        graph_dict = json.loads(graph_str)

        adj_matrix = np.array(graph_dict['adj_matrix'])
        ops = []
        for operation in graph_dict['ops']:
            op = json.loads(operation)

            # although the use of globals() is generally evil,
            # there are no caveats in this particular use-case
            # (especially since reproducibility is unlikely to be used in
            # autonomous/deployed scenarios)
            op_cls = globals()[op['op']]
            if op_cls == Network:
                # recursive deserialization to support hierarchical structures
                ops.append(Network.deserialize(operation))
            else:
                # primitive op
                ops.append(PartialWrapper(op_cls, **op['args']))

        return PartialWrapper(Network, adj_matrix, ops)

    def serialize(self):
        """
        """

        nodes = [str(op).strip() for op in self.nodes(data=True)]
        adj_matrix = self.adj_matrix.tolist()

        task_map = {}
        for k, subgraph in self.task_map.items():
            preds = {key: [o.id for o in ol] \
                     for key, ol in self.task_map[k][2].items()}
            task_map[k] = (subgraph[0].tolist(),
                           [o.id for o in subgraph[1]],
                           preds)

        params = {
            'mflops': self.flops / 1e6,
            'adj_matrix': adj_matrix,
            'nodes': nodes,
            'task_map':task_map,
            'tasks_metadata': [md.params for md in self.tasks_metadata]
        }

        return json.dumps(params, default=lambda obj: obj.params \
                          if isinstance(obj, Metadata) else str(obj))


    #---------------------------------------------------------------------------
    # UTILITIES & CUSTOM GETTERS

    def activate_task(self, task_id):
        """
        Set the active task for training

        Args:
            task_id (:obj:`Any`): the task's ID. This task must be compiled \
            prior to activation
        """
        assert task_id in self.task_map, (
            'Invalid task ID provided. Please ensure this model was compiled '
            f'for task `{task_id}` prior to activating the task sub-graph'
        )

        self.active_task = task_id

    # def get_adj_matrix(self):
    #     """[Deprecated]
    #     Returns the adjacency matrix representation of this graph \
    #     (:class:`nx.DiGraph`). This is not stored as a local variable as the \
    #     graph could be amended during compilation and so any changes to the \
    #     graph's edges need to be also be applied to the adjacency matrix. To \
    #     reduce the maintainability hassle, this is calculated on the fly.

    #     TODO: This is clearly inefficient. Ideally we would have an \
    #     intermediate class to sync the edges with the adjacency matrix so we \
    #     wouldn't have to "calculate" the adjacency matrix every time from \
    #     the edges whilst being readable and maintainable

    #     Returns:
    #         :class:`np.ndarray`: the graph's adjacency matrix representation
    #     """

    #     # ordered list of nodes as they appear in the DAG
    #     nodes = list(self.nodes())

    #     mat = np.zeros((len(nodes), len(nodes)), dtype=int)

    #     # extract connections from edges
    #     for edge in self.edges():
    #         src, dest = edge
    #         src_idx = nodes.index(src)
    #         dest_idx = nodes.index(dest)
    #         mat[src_idx, dest_idx] = 1  # diagonal

    #     return mat

    def get_subgraph(self, task_id):
        """
        Extracts the portion of the Network corresponding to the given task ID.

        Args:
            task_id (:obj:`int`): the target task ID

        Returns:
            :rtype: (:class:`np.ndarray`, :obj:`list`): the subgraph \
            pertaining to the given task, in the form of (`adjacency_matrix`, \
            `operations_list`)
        """
        assert task_id in self.task_map, (
            f'The provided task ID ({task_id}) does not have a corresponding '
            'subgraph in this Network'
        )

        return self.task_map[task_id]

    def get_torchscript(self, in_shape):
        """
        Traces the model using a random input (conforming to the same shape as
        `InputStem`) and returns a traced model object.

        Warning: slightly and memory– and computionally-expensive. Not
        recommended to use within the NAS loop

        Returns:
            :class:`torch.jit.RecursiveScriptModule`: the traced torchscript \
            model
        """

        return torch.jit.trace(self, in_shape)
                               # torch.rand(list(self.nodes()[0]).in_shape))

    def get_node_by_id(self, node_id):
        """
        Gets a node in the graph by its ID

        Args:
            node_id (:obj:`int`): the target node ID

        Returns:
            :class:`~BaseOperation`: the resulting node if it exists, \
            returns `None` otherwise
        """

        for node in self.nodes():
            if node.id == node_id:
                return node

        return None

    def get_output_layer(self, task_id):
        """
        Returns the output layer for a given task ID

        Args:
            task_id (:obj:`int`): the task ID associated with the given \
            model's sub-graph
        Return:
            :class:`nn.Module`: the output layer for the given task ID
        """

        for op in self.task_map[task_id][1]:
            if len(list(self.successors(op))) == 0:
                # no successors = output layer
                return op

        raise ValueError(f'No output layer was found for task ({task_id})')

    def clone(self):
        """
        Returns a deep copy of this network.

        Returns:
            :class:`~Network`: the cloned network
        """
        return deepcopy(self)

    def visualize(self, dir='./plots/', filename=None, show_plot=False):
        """
        Visualize the Network's graph using custom layout and positioning
        """
        # default filename
        fname = filename or f'model_{self.id}_v{self.version}.svg'

        visualize_network(self, dir=dir, filename=fname, show_plot=show_plot)


    #---------------------------------------------------------------------------
    # OPERATORS

    def is_equal(self, other, equate_isomorph=False):
        """
        An additional equality test operation with the option to account for
        isomorphism

        *Note* The :func:`__eq__` operator compares model metadata, which
        includes the `wl_hash` (accounts for isomorphism) as well as the
        trained tasks. i.e. Two graphs may be identical or isomorphic, but
        one of which may have been fine-tuned or retrained, making them
        distinct and unequal.

        Args:
            other (:class:`~Network`): the second operand in the equality test
            equate_isomorph (:obj:`bool`): whether or not to account for \
            isomoprhism in the equality

        Returns:
            :obj:`bool`: result of the comparison test
        """
        if not isinstance(other, Network):
            return False

        if equate_isomorph:
            return hash(self) == hash(other)
        else:
            return str(self) == str(other)


    def __str__(self):
        return str(self.metadata)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        """
        Cannot hash the property :func:`~Network.metadata` as it creates a
        circular dependency (weisfeiler_lehman hash also calls
        :func:`~Network.__hash__`, as well as `total_parameters` + several torch
        internal  methods). This preliminary hashing takes into account
        attributes that ensure the network's uniqueness

        Returns
            :obj:`int`: Python's hash of the network's string representation
        """

        return hash(str(self.serialize()))

    def __eq__(self, other):
        if not isinstance(other, Network):
            return False

        return self.metadata == other.metadata


