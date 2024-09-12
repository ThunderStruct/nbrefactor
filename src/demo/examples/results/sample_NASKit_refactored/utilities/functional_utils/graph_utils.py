import numpy as np


# GRAPH-RELATED UTILITIES

def predecessor_successor_lists(adj_matrix):
    """
    Get a graph's predecessor and successor lists from an adjacency matrix.

    Example:
    --------
        `adj_matrix`:
            [[0, 1, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]]

        Predecessor list (inputs of every node):
            [ [], [0], [0], [1, 2], [2, 3] ]

        Successor list (outputs of every node):
            [ [1, 2], [3], [3, 4], [4], [] ]

    Args:
        adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix
    Returns:
        tuple: a 2-element tuple consisting of a predecessor and a successor \
        list, respectively
    """

    num_vertices = adj_matrix.shape[0]

    pred = [[] for _ in range(num_vertices)]    # input connections per vertex
    succ = [[] for _ in range(num_vertices)]    # output connections per vertex

    # init input/output connections for every operation
    for src in range(num_vertices):
        for dst in range(num_vertices):
            if adj_matrix[src, dst] == 1:
                pred[dst].append(src)
                succ[src].append(dst)

    return pred, succ


def path_exists(adj_matrix, src, dst):
    """
    Uses DFS to check if a path exists from `src` to `dst` in a
    given adjacency matrix

    Args:
        adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix
        src (int): index of the source node
        dst (int): index of the destination node
    Returns:
        bool: whether or not a path was found
    """

    num_vertices = len(adj_matrix)
    vis = [False] * num_vertices    # visited array

    def dfs(node):
        vis[node] = True
        if node == dst:
            # reached dst
            return True

        for neighbor, edge in enumerate(adj_matrix[node]):
            if edge == 1 and not vis[neighbor] and dfs(neighbor):
                return True

        return False

    return dfs(src)


def get_topological_orders(adj_matrix=None, pred=None):
    """
    Computes the topological orders (depths) of each node in a graph /
    adjacency matrix. This function follows the same logic behind
    :func:`nx.topological_sort()`'s indexing.

    Note: should not be used with super-graphs (i.e. non-single-rooted DAGs)

    Args:
        adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix
    Returns:
        :obj:`list`: the depth of each node in the given graph
    """
    assert adj_matrix is not None or pred is not None, (
        'You must provide the adj_matrix or pred list to compute '
        'the topological orders'
    )

    if pred is None:
        pred, _ = predecessor_successor_lists(adj_matrix)

    depths = [-1 for _ in range(len(pred))]     # memoization list

    def depth_dfs(node):
        if depths[node] >= 0:
            # memoized
            return depths[node]

        if not pred[node]:  # no predecessors
            depths[node] = 0
        else:
            depths[node] = max(depth_dfs(prev_node) \
                               for prev_node in pred[node]) + 1
        return depths[node]

    # populate `depths`
    for op, _ in enumerate(pred):
        depth_dfs(op)

    return depths


def dag_is_rooted_tree(adj_matrix):
    """
    Checks whether or not a given DAG is:
        1. Single-rooted
        2. Has no intermediary leaf nodes (single output)
        3. Has at least 1 path from root to output

    These criteria are used to define the validity of a CNN. Multi-head models
    are explicitly defined during the generation of a network; this method is
    exclusively used to enforce topologies that are generated randomly.

    Args:
        adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix

    Returns:
        bool: whether or not the DAG is a Rooted Tree
    """

    pred, succ = predecessor_successor_lists(adj_matrix)

    if not path_exists(adj_matrix, 0, len(pred) - 1):
        # no path from root to output
        return False

    for op_idx, inputs in enumerate(pred):
        if op_idx == 0 or op_idx == len(pred) - 1:
            # skip input/output stems
            continue

        elif len(inputs) == 0 and len(succ[op_idx]) > 0:
            # secondary root detected
            return False

        elif len(inputs) > 0 and len(succ[op_idx]) == 0:
            # secondary leaf detected
            return False

    # validated
    return True


def trim_isolate_nodes(adj_matrix):
    """
    Removes island/isolate nodes (those with no neighbors; i.e. nodes with
    degree zero) from a give graph topology (adjacency matrix).

    Args:
        adj_matrix (:class:`numpy.ndarray`): the graph's adjacency matrix

    Returns:
        (:class:`numpy.ndarray`): the isolates-free `adj_matrix`
    """

    # identify neighborless nodes (rows and columns sum = 0)
    row_sums = adj_matrix.sum(axis=1)
    col_sums = adj_matrix.sum(axis=0)
    rem_indices = np.where((row_sums == 0) & (col_sums == 0))[0]

    if not rem_indices.size:
        # no isolates
        return adj_matrix

    # delete rows/columns with no edges
    ret_adj = np.delete(adj_matrix, rem_indices, axis=0)
    ret_adj = np.delete(ret_adj, rem_indices, axis=1)

    return ret_adj


