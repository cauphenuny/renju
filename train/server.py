from lib import librenju as renju
import ctypes
import signal
import sys
import subprocess

stat = [[0 for _ in range(3)] for _ in range(3)]
player1 = "bin/botzone"
player2 = "bin/botzone"

def print_stat():
    sum = stat[0][1] + stat[0][2]
    print(stat)
    if stat[0][1] > 0 and stat[0][2] > 0:
        total_base = 100 / sum
        normal_base = 100 / stat[0][1]
        reverse_base = 100 / stat[0][2]
        print("statistics:")
        print(f"player ({player1}) vs player ({player2})")
        print(f"winner: p1/p2/draw: {stat[1][0]}/{stat[2][0]}/{stat[0][0]} "
              f"({stat[1][0] * total_base:.2f}% - {stat[2][0] * total_base:.2f}%)")
        print(f"player1: win when play as 1st: {stat[1][1]}/{stat[0][1]} "
              f"({stat[1][1] * normal_base:.2f}%), 2nd: {stat[1][2]}/{stat[0][2]} "
              f"({stat[1][2] * reverse_base:.2f}%)")
        print(f"player2: win when play as 1st: {stat[2][1]}/{stat[0][2]} "
              f"({stat[2][1] * reverse_base:.2f}%), 2nd: {stat[2][2]}/{stat[0][1]} "
              f"({stat[2][2] * normal_base:.2f}%)")

def signal_handler(sig, frame):
    print('exit server')
    print_stat()
    sys.exit(0)

def run_player(player_path, game: renju.game_t):
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

def start_game(player1, player2, first_id):
    game = renju.game_new(first_id, 1000)
    players = [player1, player2]

    current_player = first_id
    while True:
        print(f"player{current_player} ({players[current_player - 1]})'s move ")
        move = run_player(players[current_player - 1], game)
        x, y = map(int, move.split())
        pos = renju.point_t(x, y)
        renju.game_add_step(ctypes.pointer(game), pos)
        renju.game_print(game)
        if (renju.is_draw(game.board)):
            print('draw')
            return 0
        if (renju.check(game.board, renju.point_t(x, y))):
            print(f'player{current_player} win')
            return current_player
        current_player = 3 - current_player

def server():
    print("A server like botzone platform, using simplified I/O.")
    # player1 = input("input first player (executable file): ")
    # player2 = input("input second player: (executable file): ")
    id = 1
    while True:
        winner = start_game(player1, player2, id)
        stat[0][id] += 1
        if winner:
            stat[winner][0] += 1
            stat[winner][(winner != id) + 1] += 1
        else:
            stat[0][0] += 1
        print_stat()
        id = 3 - id

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    renju.init()
    server()

