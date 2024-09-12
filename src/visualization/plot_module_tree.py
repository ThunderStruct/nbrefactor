
from graphviz import Digraph

def plot_module_tree(root_node, format='pdf'):
    """
    Visualize the :class:`~ModuleNode` tree structure and save it as a PDF.

    Primarily used for debugging and ensuring the generated structure is 
    as intended.
    
    Args:
        root_node (:class:`~ModuleNode`): the root node of the tree to visualize.
        output_path (str): the file path where the PDF should be saved.

    Returns:
        :class:`graphviz.Digraph`: the module tree graph to be plotted or saved
    """

    def add_nodes_edges(graph, node, parent_name=None):
        node_name = f'{id(node)}_{node.name}'   # unique node name

        graph.node(node_name, node.name)

        if parent_name:
            # create an edge from the parent to this node
            graph.edge(parent_name, node_name)

        # recursive calls to all child `~ModuleNode` nodes
        for child in node.children.values():
            add_nodes_edges(graph, child, node_name)

    dag = Digraph(comment='ModuleNode Tree')
    
    # recursively add the tree nodes to the DAG
    add_nodes_edges(dag, root_node)

    dag.format = format
    
    return dag
