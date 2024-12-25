# %%
from lib import librenju as renju
from layer import try_mps
from dataset import GomokuDataset
from predictor import Predictor
from trainer import train
from colorama import Fore
import ctypes
import time

eval_time = 1000

def eval(ctype_net, n=20):
    global eval_time
    print("start evaluating")
    renju.bind_network(ctypes.pointer(ctype_net), False)
    p1, p2 = renju.preset_players[renju.MCTS_NN], renju.preset_players[renju.MINIMAX]
    first_id = 1
    win_cnt = [0, 0, 0]
    for i in range(n):
        renju.log_disable()
        result = renju.start_game(p1, p2, first_id, eval_time, ctypes.pointer(ctype_net))
        renju.log_enable()
        win_cnt[result.winner] += 1
        renju.print_game(result.game)
        print(f'winner: {result.winner}, {win_cnt[1]}:{win_cnt[2]}')
        first_id = 3 - first_id
        if (i + 1) % (n // 5) == 0:
            print(f"finished {i + 1} games, now {win_cnt[1]}:{win_cnt[2]}")
    win_rate = win_cnt[1] * 100 / (win_cnt[1] + win_cnt[2])
    if win_rate < 30:
        color = Fore.RED
    elif win_rate < 60:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    print(f'win rate {color}{win_rate:.2f}%{Fore.RESET} ({eval_time}ms game)')
    if win_rate > 80:
        eval_time *= 2
    return win_rate

def self_play(ctype_net, dataset, n=50, time_per_step=1000):
    renju.bind_network(ctypes.pointer(ctype_net), True)
    print(f"start self-playing, time_per_step: {time_per_step}ms")
    p1, p2 = renju.preset_players[renju.MCTS_NN], renju.preset_players[renju.MCTS_NN]
    first_id = 1
    start_time = time.time()
    for i in range(n):
        if i != 0:
            renju.log_disable()
        result = renju.start_game(p1, p2, first_id, time_per_step, ctypes.pointer(ctype_net))
        renju.add_games(dataset, ctypes.pointer(result), 1)
        if i != 0:
            renju.log_enable()
        first_id = 3 - first_id
        if (i + 1) % (n // 10) == 0:
            end_time = time.time()
            average_time = (end_time - start_time) / (n // 10)
            start_time = end_time
            print(f"finished {i + 1} games, average time: {average_time:.2f} seconds, now {dataset.contents.size} samples, next_pos: {dataset.contents.next_pos}")
            print(f"last game:")
            renju.print_game(result.game)
    renju.shuffle_dataset(dataset)

# %%
if __name__ == "__main__":
    renju.init()
    dataset = GomokuDataset(file="data/5000ms.dat", device=try_mps())
    test_dataset = GomokuDataset(file="data/3000ms-raw.dat", device=try_mps())
    predictor = Predictor()
    predictor.load("model/start")
    
    games = 50
    sum = 0
    best_win_rate = 0
    update_cnt = 0
    try: 
        while True:
            ctype_net = predictor.to_ctype()
            self_play(ctype_net, dataset.handle, n=games)
            dataset.refresh()
            train(predictor, dataset, test_dataset, 2, 0.002)
            sum += games
            if sum % 250 == 0:
                predictor.save(f"model/selfplay/{sum}")
            if sum % 500 == 0:
                cur_win_rate = eval(ctype_net, n=50)
                if cur_win_rate > best_win_rate:
                    print(f"{Fore.GREEN}best model found!{Fore.RESET}")
                    predictor.save(f"model/selfplay/best")
                    update_cnt += 1
    except KeyboardInterrupt:
        dataset.save("data/selfplay1000ms.dat")
        print(f"interrupted, now {sum} games played, updated {update_cnt} model")
