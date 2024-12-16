# %%
from lib import libgomoku as gomoku
from torch import nn
from predictor import Predictor, GomokuDataset, train, try_mps
from colorama import Fore
import ctypes
import time

eval_time = 500

def eval(ctype_net, n=20):
    global eval_time
    print("start evaluating")
    gomoku.bind_network(ctypes.pointer(ctype_net), False)
    p1, p2 = gomoku.preset_players[gomoku.MCTS_NN], gomoku.preset_players[gomoku.MINIMAX]
    first_id = 1
    win_cnt = [0, 0, 0]
    for i in range(n):
        gomoku.log_disable()
        result = gomoku.start_game(p1, p2, first_id, eval_time, ctypes.pointer(ctype_net))
        gomoku.log_enable()
        win_cnt[result.winner] += 1
        gomoku.print_game(result.game)
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

def self_play(net, ctype_net=None, n=100, batch_size=32, time_per_step=100, test_iter=None):
    if ctype_net == None:
        ctype_net = net.to_ctype()
    dataset = gomoku.new_dataset(65536)
    dataset_ptr = ctypes.pointer(dataset)
    gomoku.bind_network(ctypes.pointer(ctype_net), True)
    print(f"start self-playing, time_per_step: {time_per_step}ms")
    p1, p2 = gomoku.preset_players[gomoku.MCTS_NN], gomoku.preset_players[gomoku.MCTS_NN]
    first_id = 1
    gomoku.log_disable()
    start_time = time.time()
    for i in range(n):
        result = gomoku.start_game(p1, p2, first_id, time_per_step, ctypes.pointer(ctype_net))
        gomoku.add_games(dataset_ptr, ctypes.pointer(result), 1)
        first_id = 3 - first_id
        if (i + 1) % (n // 5) == 0:
            end_time = time.time()
            average_time = (end_time - start_time) / (n // 5)
            start_time = end_time
            print(f"finished {i + 1} games, average time: {average_time:.2f} seconds, now {dataset.size} samples")
    gomoku.log_enable()
    gomoku.shuffle_dataset(dataset_ptr)
    train_iter = GomokuDataset(dataset=dataset, batch_size=batch_size)
    metrics = train(net, train_iter, test_iter, 10, 0.003, try_mps())
    return metrics

# %%
if __name__ == "__main__":
    gomoku.init()
    test_iter = GomokuDataset(file="data/3000ms-raw.dat", batch_size=256, device=try_mps())
    net = Predictor(128)
    net.load("model/current")
# %%
    play_games = 400
    sum = 0
    tim = 500
    # ctype_net = net.to_ctype()
    ctype_net = gomoku.network_t()
    gomoku.load_network(ctypes.pointer(ctype_net), "model/static.v5.128ch.mod")
    input('press enter to conitnue')
    eval(ctype_net)
    input('press enter to conitnue')
    while True:
        for i in range(0, 5):
            self_play(net, ctype_net, n=play_games, time_per_step=tim, test_iter=test_iter)
            ctype_net = net.to_ctype()
            eval(ctype_net)
            sum += play_games
            net.save(f"model/selfplay/{sum}")
            gomoku.save_network(ctypes.pointer(ctype_net), f"model/selfplay/{sum}")
        # win_rate = eval(ctype_net)
        tim *= 2


# %%


# %%
