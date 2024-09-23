import lib.gomoku as gomoku
import ctypes
import signal
import sys

nullptr = ctypes.POINTER(ctypes.c_int)()

gomoku.init()

def start_game(player1, player2, first_player): 
    game = gomoku.new_game(first_player)
    gomoku.game_print(game)
    while True:
        id = game.cur_player
        pos = gomoku.move(player1 if id == 1 else player2, nullptr, game)
        gomoku.game_add_step(ctypes.pointer(game), pos)
        gomoku.game_print(game)
        if gomoku.check_draw(game.board):
            print('draw')
            return game
        if gomoku.check(game.board, pos):
            print('player %d win' % id)
            return game

def signal_handler(sig, frame):
    print('exited trainer')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
start_game(gomoku.MCTS, gomoku.MCTS, 1)