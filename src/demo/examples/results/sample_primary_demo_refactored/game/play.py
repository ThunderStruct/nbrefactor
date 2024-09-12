from .minimax import minimax
from .visualize import init_game

def play_game():
    
    game = init_game(plot=False)
    
    game.print_board_nums()

    while game.empty_squares():
        if game.num_empty_squares() % 2 == 0:
            square = minimax(None, 0, True, game)['position']
        else:
            valid_square = False
            while not valid_square:
                square = input('Choose a move (0-8): ')
                try:
                    square = int(square)
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                except ValueError:
                    print("Invalid square. Try again.")

        if game.make_move(square, 'O' if game.num_empty_squares() % 2 == 0 else 'X'):
            game.print_board()
            print('')

            if game.current_winner:
                print(f'{game.current_winner} wins!')
                return
    print("It's a tie!")


play_game()




