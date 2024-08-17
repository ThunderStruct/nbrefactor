
from graphviz import Digraph
from ..fileops import ensure_dir

def visualize_module_tree(root_node, output_path='./module_tree.pdf'):
    """
    Visualize the :class:`~ModuleNode` tree structure and save it as a PDF.
    
    Args:
        root_node (:class:`~ModuleNode`): The root node of the tree to visualize.
        output_path (str): The file path where the PDF should be saved.
    """

    def add_nodes_edges(graph, node, parent_name=None):
        node_name = f'{id(node)}_{node.name}'   # unique node name

        graph.node(node_name, node.name)

        if parent_name:
            # create an edge from the parent to this node
            graph.edge(parent_name, node_name)

        # recursive calls to all children nodes
        for child in node.children.values():
            add_nodes_edges(graph, child, node_name)

    dag = Digraph(comment='ModuleNode Tree')
    
    # recursively add the tree nodes to the DAG
    add_nodes_edges(dag, root_node)

    # ensure existence of the given output_path
    ensure_dir(output_path)

    # render to pdf
    dag.format = 'pdf'
    dag.render(output_path, cleanup=True)
