from collections import defaultdict
from copy import copy
from copy import deepcopy
import math
from ..logger import Logger

def uniform_operation_weights(op_weights_dict,
                              topological_order, max_depth, prev_ops,
                              all_sampled_ops, in_shape, *args, **kwargs):
    """
    The default hook used for operation-sampling. This function precedes other
    custom-registered hooks.

    Assign uniform probabilities across operation-groups (rather than
    across all partials)

    i.e. for 8 instances of Conv2D configurations, we consider them
    as 1 operation and assign uniform probabilities within that group.
    This reduces bias towards operations with a large number of
    configurations

    Args:
        op_weights_dict (:obj:`dict`): a dict of the valid partial operations \
        and their corresponding weights (calculated from previous hooks if \
        applicable). The values in this dict are not guaranteed to sum to `1.0`.
        topological_order (:obj:`int`): the current node's depth in the graph. \
        Could be used to add weight for certain operations in deeper sections \
        of the architecture
        max_depth (:obj:`int`): max depth of the given graph
        prev_ops (:obj:`list`): list the current node's direct predecessor(s)
        all_sampled_ops (:obj:`list`): list of tuples of all sampled \
        operations thus far and their corresponding topological orders
        in_shape (:obj:`tuple`): the expected input shape to this node

    Returns:
        :obj:`tuple`: a tuple containing the set of operations (possibly \
        modified) and the computed weights' dict (`partial_op`:`float weight`)
    """

    # Logger.debug('Hook: \n',
    #              # f'op_set: {operation_set}\n',
    #              f'weights: {weights}\n',
    #              f'topological_order: {topological_order}\n',
    #              f'max_depth: {max_depth}\n',
    #              f'prev_ops: {prev_ops}\n',
    #              f'in_shape: {in_shape}\n')

    func_groups = defaultdict(list)
    for op_partial in op_weights_dict.keys():
        func_groups[op_partial.func].append(op_partial)

    op_groups = len(func_groups)
    op_group_weight = 1 / op_groups

    ret_w = {}

    for op_cls, hyperparam_combs in func_groups.items():
        partial_weight = op_group_weight / len(hyperparam_combs)
        for op in hyperparam_combs:
            ret_w[op] = partial_weight

    return ret_w


def incentivize_operation_overlap_hook_factory(overlapping_ops):
    """
    A factory function that returns a hook that encourages the selection of
    operations from `overlapping_ops`.

    This is the default method to create overlaps in Task-Incremental Learning
    scenarios.

    Args:
        overlapping_ops (:obj:`list`): a list of tuples comprising of 1. the \
        set of operations' partials used to compose a hook function to \
        incentivize intersecting with and 2. their corresponding topological \
        orders in the supergraph

    Returns:
        :obj:`callable`: constructed function to incentivize the given \
        operation set
    """

    # decompose the incentive set and the corresponding topological orders
    _incentive_set = {copy(t[0]) for t in overlapping_ops}
    _depths = [t[1] for t in overlapping_ops]

    def _hook(op_weights_dict, topological_order, max_depth,
              prev_ops, all_sampled_ops, in_shape, *args, **kwargs):

        min_overlap_depth = 1
        sampled_ops_ids = []

        for s_idx, (s_op, s_depth) in enumerate(all_sampled_ops):
            # check currently used operations to ensure acyclic sampling
            if 'id' not in s_op.keywords:
                continue

            sampled_ops_ids.append(s_op.keywords['id'])

            for i_op, i_depth in zip(_incentive_set, _depths):
                if s_op.keywords['id'] == i_op.keywords['id']:
                    # overlapping operation; update min_overlap_depth to prevent
                    # cycles
                    min_overlap_depth = max(min_overlap_depth, s_depth)

        ret_weights = deepcopy(op_weights_dict) # preserve referenced weights

        # incentivize weight by sqrt(n)
        incentive_factor = math.sqrt(len(ret_weights)) * 10

        for op, op_weight in op_weights_dict.items():
            for i_op, i_depth in zip(_incentive_set, _depths):
                # filter by depth to prevent cycles
                if i_depth >= min_overlap_depth:
                    # more than or equal as we can use other nodes from the
                    # same topological level

                    # test equality for intersecting keys in the partial
                    # wrapper (this mimics the equality of `op.signature`)
                    if op.intersect_equals(i_op) and \
                    i_op.keywords['id'] not in sampled_ops_ids:

                        i_op_id = i_op.keywords['id']
                        # change op ID (replace dict key) and add incentive
                        new_op = deepcopy(op)
                        Logger.debug(ret_weights.keys())
                        new_op.keywords['id'] = i_op.keywords['id']
                        old_w = ret_weights[op]
                        ret_weights[new_op] = old_w * incentive_factor
                        del ret_weights[op]

        return ret_weights

    return _hook


def spatial_ops_boost_hook(op_weights_dict,
                           topological_order, max_depth, prev_ops,
                           all_sampled_ops, in_shape, *args, **kwargs):
    """
    An optional hook to boost spatial operations' probability when the previous
    operation(s) was non-spatial.

    Args:
        op_weights_dict (:obj:`dict`): a dict of the valid partial operations \
        and their corresponding weights (calculated from previous hooks if \
        applicable). The values in this dict are not guaranteed to sum to `1.0`.
        topological_order (:obj:`int`): the current node's depth in the graph. \
        Could be used to add weight for certain operations in deeper sections \
        of the architecture
        max_depth (:obj:`int`): max depth of the given graph
        prev_ops (:obj:`list`): list the current node's direct predecessor(s)
        all_sampled_ops (:obj:`list`): list of tuples of all sampled \
        operations thus far and their corresponding topological orders
        in_shape (:obj:`tuple`): the expected input shape to this node

    Returns:
        :obj:`tuple`: a tuple containing the set of operations (possibly \
        modified) and the computed weights' dict (`partial_op`:`float weight`)
    """

    ret_w = deepcopy(op_weights_dict)

    if not any([True for op in prev_ops \
                if op.func.OPERATION_TYPE == OperationType.SPATIAL_OP]):
        # no spatial ops preceding; boost spatial probs
        incentive_factor = math.sqrt(len(ret_w)) * 10

        for op, weight in ret_w.items():
            if op.func.OPERATION_TYPE == OperationType.SPATIAL_OP:
                ret_w[op] = weight * incentive_factor

    return ret_w


