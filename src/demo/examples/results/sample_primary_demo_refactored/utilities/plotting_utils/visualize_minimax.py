from matplotlib import plt as plt
import networkx as nx


def visualize_minimax_tree(G, root, max_depth=2):
    def get_node_depth(node):   # FYI: local-scope declarations are not tracked through the CDA!
        try:
            return nx.shortest_path_length(G, source=root, target=node)
        except nx.NetworkXNoPath:
            return float('inf')

    def get_move_label(parent_board, child_board):
        for i in range(len(parent_board)):
            if parent_board[i] != child_board[i]:
                row = i // 3
                col = i % 3
                return f'{child_board[i]} to {row},{col}'
        return 'Start'

    # filter nodes that are within max_depth (otherwise it will take forever to plot as the tree is huge)
    nodes_within_depth = [node for node in G.nodes() if get_node_depth(node) <= max_depth]
    filtered_G = G.subgraph(nodes_within_depth)

    # abbreviate states for readability
    labels = {}
    for node in filtered_G.nodes():
        if node == root:
            labels[node] = 'Start'
        else:
            parent = list(filtered_G.predecessors(node))[0]
            labels[node] = get_move_label(parent, node)


    pos = nx.nx_agraph.graphviz_layout(filtered_G, prog='dot')


    plt.figure(figsize=(12, 12))
    nx.draw(filtered_G, pos, labels=labels, with_labels=True, node_size=150, node_color="lightblue", font_size=8, font_color="black")
    plt.show()
    
    

