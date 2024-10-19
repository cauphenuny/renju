from lib import libgomoku as gomoku
import ctypes
import signal
import sys
import subprocess

def signal_handler(sig, frame):
    print('exit server')
    sys.exit(0)


def run_player(player_path, game: gomoku.game_t):
    process = subprocess.Popen(
        [player_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    n = (game.count // 2) + 1
    if (game.count % 2) == 0: 
        points = [(-1, -1)]
    else:
        points = []
    points += [(game.steps[i].x, game.steps[i].y) for i in range(game.count)]
    input_data = f"{n}\n" + "\n".join(f"{x} {y}" for x, y in points) + "\n"
    stdout, stderr = process.communicate(input=input_data)
    if stderr:
        print(f"Error running {player_path}: {stderr}")
    return stdout.strip().split('\n')[0]

def server():
    print("A server like botzone platform, using simplified I/O.")
    player1_path = input("input first player (executable file): ")
    player2_path = input("input second player: (executable file): ")
    # player1_path = 'bin/botzone'
    # player2_path = 'bin/botzone'
    game = gomoku.game_new(1, 1000)
    players = [player1_path, player2_path]

    current_player = 1
    while True:
        print(f"player{current_player} ({players[current_player - 1]})'s move ")
        move = run_player(players[current_player - 1], game)
        x, y = map(int, move.split())
        pos = gomoku.to_point(x, y)
        gomoku.game_add_step(ctypes.pointer(game), pos)
        gomoku.game_print(game)
        if (gomoku.is_draw(game.board)):
            print('draw')
            return
        if (gomoku.check(game.board, gomoku.point_t(x, y))):
            print(f'player{current_player} win')
            return
        current_player = 3 - current_player

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    gomoku.init()
    server()