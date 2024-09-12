from ..utilities.plotting_utils.visualize_minimax import visualize_minimax_tree
import networkx as nx
from .tic_tac_toe import TicTacToe


class MinimaxTreeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_tree(self, game, depth=0, maximizing_player=True):
        self.root = tuple(game.board)
        self._build_tree_recursive(game, depth, maximizing_player, self.root)

    def _build_tree_recursive(self, game, depth, maximizing_player, parent_node):
        if game.current_winner or not game.empty_squares():
            return
        for move in game.available_moves():
            game.make_move(move, 'O' if maximizing_player else 'X')
            child_node = tuple(game.board)
            self.graph.add_edge(parent_node, child_node)
            self._build_tree_recursive(game, depth + 1, not maximizing_player, child_node)
            game.board[move] = ' '
            game.current_winner = None

    def plot_tree(self, max_depth=2):
        visualize_minimax_tree(self.graph, self.root, max_depth)
        
        


# this cell will be appended to the previous module

def init_game(plot=False):
    
    game = TicTacToe()
    
    if plot:
        visualizer = MinimaxTreeVisualizer()
        visualizer.build_tree(game)
        visualizer.plot_tree()
        
    return game
    

