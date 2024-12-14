# %%
from lib import libgomoku as gomoku
import ctypes
import torch
import random

def to_input_tensor(sample_input: gomoku.sample_input_t, device = torch.device('cpu')):
    p1_channel = torch.tensor(sample_input.p1_pieces, dtype=torch.float32)
    p2_channel = torch.tensor(sample_input.p2_pieces, dtype=torch.float32)
    cur_channel = torch.fill(torch.zeros(225), sample_input.current_player).reshape(15, 15)
    last_move_channel = torch.zeros([15, 15], dtype=torch.float32)
    last_move_channel[sample_input.last_move.x][sample_input.last_move.y] = 1
    return torch.stack([p1_channel, p2_channel, cur_channel, last_move_channel]).to(device)

def to_tensor(sample: gomoku.sample_t, device = torch.device('cpu')):
    X = to_input_tensor(sample.input)
    prob = torch.tensor(sample.output.prob, dtype=torch.float32).reshape(225)
    eval = torch.zeros([3], dtype=torch.float32)
    eval[sample.output.result + 1] = 1 # {-1, 0, 1} => {0, 1, 2}
    # gomoku.print_sample(sample)
    return X.to(device), prob.to(device), eval.to(device)

# %%
class GomokuDataset():
    def __init__(self, file=None, dataset=None, batch_size=128, device=torch.device('cpu')):
        if dataset == None:
            self.dataset = gomoku.dataset_t()
            self.handle = ctypes.pointer(self.dataset)
            gomoku.load_dataset(self.handle, file)
        else:
            self.dataset = dataset
            self.handle = ctypes.pointer(self.dataset)
        self.batch_size = batch_size
        self.index = 0
        self.samples = list(range(0, self.dataset.size))
        self.device = device
        random.shuffle(self.samples)
        print(f'loaded dataset {file} ({len(self.samples)} samples) with batch_size={batch_size}')

    def __del__(self):
        gomoku.free_dataset(self.handle)

    def __len__(self):
        return len(self.samples) // self.batch_size
    
    def __next__(self):
        if self.index + self.batch_size > len(self.samples):
            self.index = 0
            raise StopIteration
        X_array  = []
        prob_array  = []
        eval_array = []
        for i in self.samples[self.index : self.index + self.batch_size]:
            X, prob, eval = to_tensor(gomoku.find_sample(self.handle, i), self.device)
            X_array.append(X)
            prob_array.append(prob)
            eval_array.append(eval)
        self.index += self.batch_size
        return torch.stack(X_array), torch.stack(prob_array), torch.stack(eval_array)

    def __iter__(self):
        self.index = 0
        while self.index + self.batch_size <= len(self.samples):
            yield next(self)