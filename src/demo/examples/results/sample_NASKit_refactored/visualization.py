import textwrap
from .utilities.functional_utils.file_utils import ensure_dir
import os
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import PathPatch
from .utilities.config import Config
from matplotlib.patches import Circle
from matplotlib import plt as plt
import networkx as nx

def visualize_network(graph, filename=None,
                      dir='./plots/', show_plot=True):

    assert len(graph.nodes()) > 0, (
        'The network must be compiled prior to visualization'
    )

    if filename is None and not show_plot:
        # not saving or showing the plot
        return

    # macros
    node_radius = 0.04
    box_height = 0.05
    box_width = 0.225
    text_pad = 0.01  # padding inside the box

    # Step 1: assign hierarchical levels
    def assign_levels(G):
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            level = 0 if not preds else max(levels[pred] for pred in preds) + 1
            levels[node] = level
        return levels

    # Step 2: calculate node positions (both x & y)
    def calculate_positions(graph, levels, x_spacing_factor, y_spacing_factor):
        positions = {}
        max_depth = max(levels.values())

        level_heights = {}

        for node, level in levels.items():
            if level not in level_heights:
                level_heights[level] = 0
            level_heights[level] += 1

        level_idx = {level: 0 for level in level_heights}

        for node in graph.nodes():
            level = levels[node]
            x_pos = level / max_depth

            y_pos = (level_idx[level] + 0.5) / level_heights[level]
            level_idx[level] += 1
            positions[node] = (x_pos * x_spacing_factor,
                               y_pos * y_spacing_factor)

        return positions


    # Step 3: draw the network
    def draw_neural_network(G, ax, legend_names=False):
        levels = assign_levels(graph)
        pos = calculate_positions(graph, levels, 1.25, 1.25)

        nodes_per_level = {}
        for node, level in levels.items():
            if level not in nodes_per_level:
                nodes_per_level[level] = 0
            nodes_per_level[level] += 1

        # plot limits calc
        all_x = [x for x, y in pos.values()]
        all_y = [y for x, y in pos.values()]

        margin_x = 0.25  # margin outermost nodes for a bit of space
        margin_y = 0.30

        ax.set_xlim(min(all_x) - margin_x,
                    max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y,
                    max(all_y) + margin_y)

        # draw edges (with a low z-order so they'd be covered by the nodes)
        for edge in G.edges():
            # x0, y0 = pos[edge[0]]
            # x1, y1 = pos[edge[1]]

            # ax.plot([x0, x1], [y0, y1], 'lightgrey', lw=1, zorder=1)

            start, end = pos[edge[0]], pos[edge[1]]
            level_diff = abs(levels[edge[0]] - levels[edge[1]])

            # number of nodes at each level
            diff_nodes_lvl = nodes_per_level[levels[edge[0]]] \
            != nodes_per_level[levels[edge[1]]]

            if level_diff > 1 and not diff_nodes_lvl:
                # curved edges for non-adjacent levels / same node counts
                control = [(start[0] + end[0]) / 2,
                           (start[1] + end[1]) / 2 + 0.1 * level_diff]
                path = Path([start, control, end],
                            [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                patch = PathPatch(path, facecolor='none', lw=1,
                                  edgecolor='lightgrey', zorder=1)
                ax.add_patch(patch)
            else:
                # straight edge path otherwise
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color='lightgrey', lw=1, zorder=1)


        for node, (x, y) in pos.items():
            # draw the node circle
            circle = Circle(pos[node], radius=node_radius, edgecolor='grey',
                            facecolor=node.op_color, lw=2, alpha=1.0, zorder=3)
            ax.add_patch(circle)
            # layer IDs in the middle of the node
            plt.text(pos[node][0], pos[node][1], str(node.id),
                     ha='center', va='center', color='white', fontsize=8)

            if not legend_names:
                # draw text box
                rect = plt.Rectangle((x - box_width / 2,
                                    y - node_radius - box_height - text_pad),
                                    box_width, box_height,
                                    linewidth=1, edgecolor='black',
                                    facecolor='grey', alpha=0.8, zorder=4)
                ax.add_patch(rect)
                ax.text(x, y - node_radius - box_height / 2 - text_pad,
                        node.op_name, ha='center', va='center',
                        color='white', fontsize=6, zorder=5)
            else:
                # draw legend (mapping IDs to layer names)
                legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                             label=f'{node.id}: {node.op_name}',
                                             markerfacecolor=node.op_color,
                                             markersize=10,
                                             alpha=0.75) \
                                  for node in graph.nodes()]
                ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
                          loc='upper left')

        version_text = f' -- v{graph.version}' if graph.version > 1 else ''
        plt.text(x=0.025, y=0.975,
                 s=f'MODEL ID ({graph.id}{version_text})',
                 transform=plt.gca().transAxes, fontsize=5, color='grey',
                 verticalalignment='top', style='italic')


    # draw
    fig, ax = plt.subplots(figsize=(8, 8))
    # use legend_names by default as it is cleaner visually
    draw_neural_network(graph, ax, legend_names=False)


    ax.axis('off')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()

    if filename is not None:
        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        plt.savefig(full_path)

    if show_plot:
        plt.show()




""" DEPRECATED """

def __visualize_network(graph, dir='./plots/', filename=None, show_plot=True):

    assert len(graph.nodes()) > 0, (
        'The network must be compiled prior to visualization'
    )

    if filename is None and not show_plot:
        # not saving or showing the plot
        return

    adj_matrix = graph.adj_matrix
    num_nodes = adj_matrix.shape[0]

    # --== start sub-methods
    def nodes_pos():
        pos = {}
        max_depth = max(nx.get_node_attributes(graph, 'depth').values())
        for depth in range(max_depth + 1):
            nodes_at_depth = [node[0] for node in graph.nodes(data=True) \
                              if node[1]['depth'] == depth]
            max_width = max([len(node.op_name) for node in nodes_at_depth])
            for i, node in enumerate(nodes_at_depth):
                y_pos = i - (len(nodes_at_depth) - 1) / 2
                # symmetrical across the x-axis
                pos[node] = (-(max_depth * 2 - depth) * 1.2, y_pos)

        return pos

    def draw_nodes(pos, ax, **kwargs):
        path = None
        for node in graph.nodes():
            # label = '\n'.join(textwrap.wrap(node.op_name, 15))
            # node_x, node_y = pos[node]
            # rect = FancyBboxPatch((node_x - 0.45, node_y - 0.075), 0.9, 0.15,
            #                       boxstyle='round,pad=0.02',
            #                       ec='black', fc=node.op_color, alpha=0.4)
            # # rect = plt.Rectangle((node_x - 0.45, node_y - 0.075), 0.9, 0.15,
            # #                      facecolor='#fff', edgecolor=node.op_color,
            # #                      **kwargs)
            # ax.add_patch(rect)
            # ax.text(node_x, node_y, label, fontsize=8, ha='center',
            #         va='center')
            x, y = pos[node]
            label = '\n'.join(textwrap.wrap(node.op_name, 20))
            text_obj = ax.text(x, y, label, fontsize=8,
                               ha='center', va='center')

            renderer = fig.canvas.get_renderer()
            text_width = text_obj.get_window_extent(renderer).width

            bbox_width = text_width / fig.dpi * 0.9
            bbox_height = 0.1
            text_obj.remove()
            rect = FancyBboxPatch((x - bbox_width / 2, y - bbox_height / 2),
                                  bbox_width, bbox_height,
                                boxstyle='round,pad=0.02', ec='black',
                                  fc=node.op_color, alpha=0.4)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=8, ha='center', va='center')
            if path is None:
                path = rect.get_path()

        return path
    # --== end sub-methods

    pred = [list(graph.successors(node)) for node in graph.nodes()]

    # # remove isolates
    # for idx, node_conns in enumerate(input_conns):
    #     if len(node_conns) == 0 and idx > 0:
    #         graph.remove_node(idx)


    # Logger.debug(graph.nodes(), '\n', graph.adj_matrix, '\n',
    #              graph.compiled_ops)
    # node_labels = {node: node for node in graph.nodes()}
    # node_shapes = {node: 's' for node in graph.nodes()}

    custom_nn_pos = nodes_pos()
    # draw
    fig, ax = plt.subplots(figsize=(15, 10))

    nx.draw_networkx_edges(graph, custom_nn_pos, ax=ax, node_shape='s',
                           node_size=10, alpha=0.25)
                            #, connectionstyle='arc3, rad=0.05')
    draw_path = draw_nodes(custom_nn_pos, ax, linewidth=1.5, alpha=1.0)

    ax.axis('off')

    if filename is not None:
        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        plt.savefig(full_path)

    if show_plot:
        plt.show()


    # nx.draw(self, pos=custom_nn_pos, with_labels=False,
    #         node_size=500, node_color='lightblue',
    #         font_size=8, font_color='black', node_shape='s', alpha=0.9)

    # nx.draw_networkx_labels(self, custom_nn_pos,
    #                         labels=node_labels, font_size=8,
    #                         font_color='black', verticalalignment='bottom',
    #                         horizontalalignment='center')



# def alt_visualize(G, node_labels=None):
#     """
#     Draw a customized representation of a neural network with consistent
#     spacing between nodes.

#     Args:
#     layers (list of int): A list containing the number of nodes in each layer.
#     node_labels (dict): Optional. A dictionary with node index as keys and \
#                         labels as values.
#                         If None, nodes are labeled with their indices.
#     """

#     layers = list(G.nodes())
#     total_nodes = len(layers)
#     pos = {}
#     layer_positions = [len(layers[:i]) for i in range(len(layers) + 1)]
#     bbox_sizes = {}
#     # Dummy figure to calculate text sizes
#     fig, ax = plt.subplots()
#     for i, layer in enumerate(layers):
#         max_width = 0
#         for j in range(layer):
#             node = layer_positions[i] + j
#             G.add_node(node)
#             label = node.op_name
#             text_size = ax.text(x, y, label, fontsize=8,
#                                 ha='center', va='center')
#             renderer = fig.canvas.get_renderer()
#             text_width = text_obj.get_window_extent(renderer).width
#             text_height = text_obj.get_window_extent(renderer).height

#             bbox_width = text_width / fig.dpi * 0.9
#             bbox_height = text_height / fig.dpi * 0.75
#             bbox_sizes[node] = (bbox_width, bbox_height)
#             max_width = max(max_width, bbox_width)
#         for j in range(layer):
#             node = layer_positions[i] + j
#             pos[node] = (i * (max_width + 0.2), -j * \
#                          (bbox_sizes[node][1] + 0.2))
#     plt.close(fig)

#     # Create edges
#     for i in range(len(layers) - 1):
#         for j in range(layers[i]):
#             for k in range(layers[i + 1]):
#                 G.add_edge(layer_positions[i] + j, layer_positions[i + 1] + k)

#     # Draw the network with spacing
#     fig, ax = plt.subplots()
#     nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
#     for node in G.nodes:
#         x, y = pos[node]
#         label = str(node_labels[node]) \
#                 if node_labels and node in node_labels else str(node)
#         bbox_width, bbox_height = bbox_sizes[node]
#         bbox = FancyBboxPatch((x - bbox_width / 2, y - bbox_height / 2),
#                               bbox_width, bbox_height,
#                               boxstyle="round,pad=0.02", ec="black",
#                               fc=node.op_color, alpha=0.7)
#         ax.add_patch(bbox)
#         ax.text(x, y, label, fontsize=8, ha='center', va='center')

#     plt.title("Artificial Neural Network Architecture")
#     plt.axis("off")
#     plt.show()


