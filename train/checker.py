#!/usr/bin/env python3
# author: Cauphenuny <https://cauphenuny.github.io/>

from IPython import display
from lib import libgomoku as gomoku
import random
import sys
import torch
from d2l import torch as d2l
from torch import nn
import ctypes

class GomokuDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, start, end):
        self.samples = list(range(start, end))
        random.shuffle(self.samples)
        self.batch_size = batch_size
        self.index = 0
        

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        self.index = 0
        while self.index + self.batch_size < len(self.samples):
            X = torch.stack([torch.tensor(gomoku.find_sample(i).board, dtype=torch.float32).reshape(1, 15, 15) for i in self.samples[self.index:self.index+self.batch_size]])
            y = torch.stack([torch.tensor(gomoku.find_sample(i).winner, dtype=torch.float32) for i in self.samples[self.index:self.index+self.batch_size]])
            self.index += self.batch_size
            yield X, y

    def __next__(self):
        if self.index + self.batch_size > len(self.samples):
            raise StopIteration
        X = torch.stack([torch.tensor(gomoku.find_sample(i).board, dtype=torch.float32).reshape(1, 15, 15) for i in self.samples[self.index:self.index+self.batch_size]])
        y = torch.stack([torch.tensor(gomoku.find_sample(i).winner, dtype=torch.float32) for i in self.samples[self.index:self.index+self.batch_size]])
        self.index += self.batch_size
        return X, y

def try_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

test_size = 1000

def test_iter(batch_size):
    return GomokuDataset(batch_size, 0, test_size)

def train_iter(batch_size):
    return GomokuDataset(batch_size, test_size, gomoku.dataset_size())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    # net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 1, 1], dtype=torch.float32).to(device))
    timer, num_batches = d2l.Timer(), len(train_iter)
    print(f'num_batches: {num_batches}')
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

class Checker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(), # 8 * 15 * 15
            nn.MaxPool2d(kernel_size=5, stride=5),                # 8 * 3 * 3
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(288, 3)
        )
        self.conv = {}
        self.linear = {}

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def forward(self, X):
        return self.net(X)

    def to_ctype(self):
        ctype_net = gomoku.checker_network_t()
        ctype_net.conv.weight = (ctypes.c_float * 800)()
        ctype_net.conv.bias = (ctypes.c_float * 32)()
        ctype_net.linear.weight = (ctypes.c_float * 864)()
        ctype_net.linear.bias = (ctypes.c_float * 3)()
        print(ctype_net)
        for ch in range(32):
            for i in range(5):
                for j in range(5):
                    ctype_net.conv.weight[ch * 25 + j * 5 + i] = self.net[0].weight[ch][0][j][i].item()
        for i in range(32):
            ctype_net.conv.bias[i] = self.net[0].bias[i].item()
        for i in range(3):
            for j in range(288):
                ctype_net.linear.weight[i * 288 + j] = self.net[5].weight[i][j].item()
        for i in range(3):
            ctype_net.linear.bias[i] = self.net[5].bias[i].item()
        return ctype_net

checker = Checker()

def test():
    checker.eval()
    first_id = 1
    game: gomoku.game_t = gomoku.game_new(first_id, 1000)
    players = [gomoku.preset_players[gomoku.MCTS], gomoku.preset_players[gomoku.MCTS]]

    current_player = first_id
    while True:
        print(f"player{current_player} ({players[current_player - 1]})'s move ")
        player = players[current_player - 1]
        pos = gomoku.move(game, player)
        gomoku.game_add_step(ctypes.pointer(game), pos)
        # gomoku.game_print(game)
        transformed_board = game.board
        for i in range(15):
            for j in range(15):
                if transformed_board[i][j] == 2:
                    transformed_board[i][j] = -1
        X = torch.tensor(transformed_board, dtype=torch.float32).reshape(1, 1, 15, 15)
        y_hat = checker(X)
        ctype_net = checker.to_ctype()
        print(y_hat.argmax(axis=1)[0], y_hat.data)
        gomoku.checker_forward(ctypes.pointer(ctype_net), game.board)
        if (gomoku.is_draw(game.board)):
            print('draw')
            return 0
        if (gomoku.check(game.board, pos)):
            print(f'player{current_player} win')
            return current_player
        current_player = 3 - current_player

if __name__ == "__main__":
    gomoku.init()
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} [.dat file]')
        sys.exit(1)
    if gomoku.import_samples(sys.argv[1]):
        sys.exit(2)
    batch_size = 16
    X, y = next(train_iter(batch_size))
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    for layer in checker.net:
        print(f'layer: {layer}, parameters: {list(x.shape for x in layer.parameters())}')
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)
    
    train(checker, train_iter(batch_size), test_iter(batch_size), 15, 0.05, try_mps())
    # checker.load("model/checker_tmp.params")

    for layer in checker.net:
        print(f'layer: {layer}, parameters: {list(x for x in layer.parameters())}')

    checker.to('cpu')

    ctype_net = checker.to_ctype()
    gomoku.checker_save(ctypes.pointer(ctype_net), "model/checker_tmp.mod")

    # test()