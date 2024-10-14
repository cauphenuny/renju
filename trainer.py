import lib.gomoku as gomoku
import ctypes
import signal
import sys

nullptr = ctypes.POINTER(ctypes.c_int)()

gomoku.init()

def start_game(player1_id, player2_id, first_player): 
    game = gomoku.new_game(first_player, 1000)
    players = [gomoku.preset_players[player1_id], gomoku.preset_players[player2_id]]
    gomoku.game_print(game)
    while True:
        id = game.cur_id
        player = players[id - 1]
        pos = gomoku.move(game, player)
        gomoku.game_add_step(ctypes.pointer(game), pos)
        gomoku.game_print(game)
        if gomoku.check_draw(game.board):
            print('draw')
            return game
        if gomoku.check(game.board, pos):
            print('player %d win' % id)
            return game

def signal_handler(sig, frame):
    print('exit trainer')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
start_game(gomoku.MCTS, gomoku.MCTS, 1)
