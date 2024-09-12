import math



def minimax(state, depth, maximizing_player, game):
    if game.current_winner == 'O':
        return {'position': None, 'score': 1 * (depth + 1)}  # maximizing
    elif game.current_winner == 'X':
        return {'position': None, 'score': -1 * (depth + 1)}  # minimizing
    elif not game.empty_squares():
        return {'position': None, 'score': 0}

    if maximizing_player:
        best = {'position': None, 'score': -math.inf}
    else:
        best = {'position': None, 'score': math.inf}

    for possible_move in game.available_moves():

        game.make_move(possible_move, 'O' if maximizing_player else 'X')

        sim_score = minimax(state, depth + 1, not maximizing_player, game)

        game.board[possible_move] = ' '
        game.current_winner = None
        sim_score['position'] = possible_move


        if maximizing_player:
            if sim_score['score'] > best['score']:
                best = sim_score
        else:
            if sim_score['score'] < best['score']:
                best = sim_score

    return best

