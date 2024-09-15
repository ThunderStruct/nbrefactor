""" Visualization methods for refactored notebooks
"""

# import networkx as nx
from graphviz import Digraph
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout

def plot_module_tree(root_node, format='pdf', ):
    """
    Visualize the :class:`~nbrefactor.datastructs.ModuleNode` tree structure \
        and save it as a PDF.

    Primarily used for debugging and ensuring the generated structure is 
    as intended.
    
    Args:
        root_node (:class:`~nbrefactor.datastructs.ModuleNode`): the root \
            node of the tree to visualize.
        output_path (str): the file path where the PDF should be saved.

    Returns:
        :class:`graphviz.Digraph`: the module tree graph to be plotted or saved
    """

    def add_nodes_edges(graph, node, parent_name=None):
        node_name = f'{id(node)}_{node.name}'   # using id() to ensure
                                                # node-name uniqueness

        graph.node(node_name, node.name)

        if parent_name:
            # create an edge from the parent to this node
            graph.edge(parent_name, node_name, minlen='2')

        # recursive calls to all child `~ModuleNode` nodes
        for child in node.children.values():
            add_nodes_edges(graph, child, node_name)

    dag = Digraph(comment='ModuleNodeTree', 
                  graph_attr={'splines': 'splines',
                              'rankdir': 'TB', 
                              'nodesep': '0.5'})
    
    # recursively add the tree nodes to the DAG
    add_nodes_edges(dag, root_node)

    dag.format = format
    
    return dag



# TODO: Needs improvements. It has to look much better to justify the
# dependency clutter. Could potentially also explore an igraph + plotly 
# approach to get that *hierarchical directory tree* look; refer to
# (https://plotly.com/python/tree-plots/)

# def plot_module_tree_nx(root_node, output_path):
#     """
#     Alternative visualization method to draw the :class:`~ModuleNode` tree \
#     structure using NetworkX, Graphviz layout, and plt.

#     Args:
#         root_node (:class:`~nbrefactor.datastructs.ModuleNode`): the root \
#            node of the tree to visualize.
    
#     Returns:
#         None: the plot is displayed with matplotlib.
#     """

#     def truncate_name(name, max_length=50):
#         if len(name) <= max_length:
#             return name
#         half_length = (max_length - 3) // 2
#         return f'{name[:half_length]}...{name[-half_length:]}'
    
#     def add_edges_to_graph(graph, node):
#         node_name = f'{node.name}'

#         node_name = truncate_name(node_name)

#         for child in node.children.values():
#             child_name = truncate_name(f'{child.name}')
#             graph.add_edge(node_name, child_name)
#             add_edges_to_graph(graph, child)

#     dag = nx.DiGraph()
#     add_edges_to_graph(dag, root_node)
#     pos = graphviz_layout(dag, prog='dot')

#     plt.figure(figsize=(18, 8))

#     nx.draw(dag, pos, with_labels=True, node_size=3000, node_shape='$\u25AC$',
#             node_color='none', font_size=10, edge_color='gray')
    
#     plt.savefig(output_path)

