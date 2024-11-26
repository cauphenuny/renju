#!/usr/bin/env python3
# author: Cauphenuny <https://cauphenuny.github.io/>

from IPython import display
from lib import libgomoku as gomoku
import random
import torch
from d2l import torch as d2l
from torch import nn
import ctypes
import sys

def try_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class GomokuDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, start, end):
        self.samples = list(range(start, end))
        random.shuffle(self.samples)
        self.batch_size = batch_size
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def __next__(self):
        if self.index + self.batch_size > len(self.samples):
            raise StopIteration
        X  = []
        prob  = []
        eval = []
        for i in self.samples[self.index : self.index + self.batch_size]:
            raw_sample = gomoku.find_sample(i)
            X.append(torch.stack([torch.tensor(raw_sample.board, dtype=torch.float32), 
                                    torch.tensor(raw_sample.current_id, dtype=torch.float32)]))
            prob.append(torch.tensor(raw_sample.prob, dtype=torch.float32).reshape(1, 15, 15))
            eval.append(torch.tensor(raw_sample.result, dtype=torch.float32))
        self.index += self.batch_size
        return torch.stack(X), torch.stack(prob), torch.stack(eval)

    def __iter__(self):
        self.index = 0
        while self.index + self.batch_size <= len(self.samples):
            yield next(self)

test_size = 0

def test_iter(batch_size):
    return GomokuDataset(batch_size, 0, test_size)

def train_iter(batch_size):
    return GomokuDataset(batch_size, test_size, gomoku.dataset_size())

class Predictor(nn.Module):
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

if __name__ == '__main__':
    gomoku.init()
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} [.dat file]')
        sys.exit(1)
    if gomoku.import_samples(sys.argv[1]):
        sys.exit(2)
    test_size = gomoku.dataset_size() // 10
    print(f'dataset size: {gomoku.dataset_size()}, test size: {test_size}')
    input('press enter to continue...')

    for board, prob, eval in train_iter(1):
        print(board, '\n', prob, '\n', eval)
        input('press enter to continue...')