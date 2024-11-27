# %%
#!/usr/bin/env python3
# author: Cauphenuny <https://cauphenuny.github.io/>

from IPython import display
from colorama import Fore
from lib import libgomoku as gomoku
import random
import torch
from d2l import torch as d2l
from torch import nn
import ctypes
import sys
from export import to_ctype

def cpu():
    return torch.device("cpu")

def try_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_input_tensor(input):
    return torch.stack([torch.tensor(input.board, dtype=torch.float32), 
                        torch.tensor(input.cur_id, dtype=torch.float32)])

def to_tensor(raw_sample, device = cpu()):
    X = torch.stack([torch.tensor(raw_sample.board, dtype=torch.float32), 
                     torch.tensor(raw_sample.cur_id, dtype=torch.float32)])
    prob = torch.tensor(raw_sample.prob, dtype=torch.float32).reshape(225)
    eval = torch.tensor(raw_sample.result, dtype=torch.float32).reshape(1)
    return X.to(device), prob.to(device), eval.to(device)
    

class GomokuDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, start, end):
        self.samples = list(range(start, end))
        random.shuffle(self.samples)
        self.batch_size = batch_size
        self.index = 0
        print(f'{Fore.GREEN}initialized dataset [{start}, {end}) with batch_size={batch_size}{Fore.RESET}')

    def __len__(self):
        return len(self.samples)

    def __next__(self):
        if self.index + self.batch_size > len(self.samples):
            raise StopIteration
        input_array  = []
        prob_array  = []
        eval_array = []
        for i in self.samples[self.index : self.index + self.batch_size]:
            X, prob, eval = to_tensor(gomoku.find_sample(i))
            input_array.append(X)
            prob_array.append(prob)
            eval_array.append(eval)
        self.index += self.batch_size
        return torch.stack(input_array), torch.stack(prob_array), torch.stack(eval_array)

    def __iter__(self):
        self.index = 0
        while self.index + self.batch_size <= len(self.samples):
            yield next(self)

test_size = 0

def test_iter(batch_size):
    return GomokuDataset(batch_size, 0, test_size)

def train_iter(batch_size):
    return GomokuDataset(batch_size, test_size, gomoku.dataset_size())

def evaluate_loss(net, device, iter, loss):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, _, y in iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            metric.add(l * X.shape[0], X.shape[0])
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, _, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Predictor(nn.Module):
    def __init__(self, max_channel = 64, model_name = None):
        super().__init__()
        """
shared = 2 * 32 * 225 * 9 + 32 * 64 * 225 * 25 + 64 * 128 * 225 * 9
value = 128 * 4 * 1 + 4 * 225 * 128 + 128 * 1
policy = 128 * 32 * 225 * 9 + 16 * 1 * 225 * 1
print(f'shared: {shared}, value: {value}, policy: {policy}, sum: {(shared+value+policy)/1e6}')
        """
        self.shared = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2), nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(), 
            nn.Conv2d(64, max_channel, kernel_size=3, padding=1), nn.ReLU()
        )
        self.value_net = nn.Sequential(
            self.shared, 
            nn.Conv2d(max_channel, 4, kernel_size=1, padding=0), nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(4 * 15 * 15, 128), nn.ReLU(), nn.Dropout(), 
            nn.Linear(128, 1), nn.Tanh()
        )
        self.policy_net = nn.Sequential(
            self.shared, 
            nn.Conv2d(max_channel, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(), 
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.Flatten(), nn.LogSoftmax(dim=1)
        )
        self.max_channel = max_channel
        if model_name != None:
            self.load(model_name)
        self.model_name = model_name

    def save(self, basename):
        name = f'{basename}.v{gomoku.NETWORK_VERSION}.{self.max_channel}ch.params'
        torch.save(self.state_dict(), name)

    def load(self, basename):
        name = f'{basename}.v{gomoku.NETWORK_VERSION}.{self.max_channel}ch.params'
        self.to(try_mps())
        print(f'load params from {name}')
        self.load_state_dict(torch.load(name))

    def forward(self, X):
        return self.policy_net(X), self.value_net(X)
    
    def to_ctype(self):
        net = gomoku.predictor_network_t()
        print('created')
        net.shared.conv1.weight, net.shared.conv1.bias = to_ctype(self.shared[0])
        print('completed 1')
        net.shared.conv2.weight, net.shared.conv2.bias = to_ctype(self.shared[2])
        print('completed 2')
        net.shared.conv3.weight, net.shared.conv3.bias = to_ctype(self.shared[5])
        print('completed 3')
        net.value.conv.weight, net.value.conv.bias = to_ctype(self.value_net[1])
        print('completed 4')
        net.value.linear1.weight, net.value.linear1.bias = to_ctype(self.value_net[4])
        print('completed 5')
        net.value.linear2.weight, net.value.linear2.bias = to_ctype(self.value_net[7])
        print('completed 6')
        net.policy.conv1.weight, net.policy.conv1.bias = to_ctype(self.policy_net[1])
        print('completed 7')
        net.policy.conv2.weight, net.policy.conv2.bias = to_ctype(self.policy_net[4])
        print('completed 8')
        return net

    def export_ctype(self, name = None):
        if name == None:
            name = self.model_name
        if name == None:
            name = input('input model name: ')
        print(f'exporting model "{name}"')
        ctype_net = self.to_ctype()
        gomoku.predictor_save(ctypes.pointer(ctype_net), name)

# %%
def calculate_entropy(tensor):
    return -torch.sum(torch.exp(tensor) * tensor)

def print_sample(sample):
    # gomoku.print_sample(sample)
    X, prob, eval = to_tensor(sample)
    X = X.reshape(1, 2, 15, 15)

    print(f'eval: {eval}')
    print(f'entropy: {calculate_entropy(prob)}')
    print(f'prob detail: {prob}')

# %%
def train(animated, net, train_iter, test_iter, num_epochs, lr, device):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    timer, num_batches = d2l.Timer(), len(train_iter)
    print(f'num_batches: {num_batches}')
    loss = nn.MSELoss()
    metric_record = []
    if animated:
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 3.0],
                        legend=['train loss', 'test loss', 'entropy'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        for i, (X, policy, value) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, policy, value = X.to(device), policy.to(device), value.to(device)
            policy_hat, value_hat = net(X)
            value_loss = loss(value_hat, value)
            policy_loss = -torch.mean(torch.sum(policy * policy_hat, dim=1))
            l = value_loss + policy_loss
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(X.shape[0], 
                           value_loss * X.shape[0],
                           policy_loss * X.shape[0],
                           sum(calculate_entropy(policy_hat[i]) for i in range(X.shape[0])))
            timer.stop()
        train_l1 = metric[1] / metric[0]
        train_l2 = metric[2] / metric[0]
        entropy = metric[3] / metric[0]
        test_l1 = evaluate_loss(net.value_net, device, test_iter, loss)
        metric_record.append([train_l1, test_l1, train_l2, entropy])
        if animated:
            animator.add(epoch + 1, (train_l1, test_l1, entropy))
        else:
            print(f'epoch {epoch + 1}, loss {train_l1:.3f}, {train_l2:.3f}, test value loss {test_l1:.3f}, entropy {entropy:.3f}')
    print(f'value loss {train_l1:.3f}, policy loss {train_l2:.3f}, test value loss {test_l1:.3f}, entropy {entropy:.3f}')
    print(f'{metric[0] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    return metric_record


# %%
def test_sample(net, ctype_net, sample):
    X, prob, eval = to_tensor(sample)
    X = X.reshape(1, 2, 15, 15).to(try_mps())
    net.to(try_mps())
    net.eval()

    # print(X)

    # detail(net, X)

    log_prob_hat, eval_hat = net(X)

    prob_hat = torch.exp(log_prob_hat)

    print("probability:")
    # print(prob_hat)
    board_array = gomoku.board_t()
    prob_array = gomoku.fboard_t()
    for x in range(15):
        for y in range(15):
            board_array[x][y] = 0 if sample.board[x][y] == 0 else (1 if sample.board[x][y] == 1 else 2)
            prob_array[x][y] = prob_hat[0][x * 15 + y].item()
    print(calculate_entropy(prob_hat[0]))
    gomoku.probability_print(board_array, prob_array)

    print(f'eval: {eval_hat[0][0].item():.3f}')

    cur_id = sample.cur_id[0][0]
    if cur_id == -1:
        cur_id = 2
    print('call gomoku.predict...')
    prediction = gomoku.predict(ctypes.pointer(ctype_net), board_array, 1, 0)
    print(f'result eval: {prediction.eval}')
    gomoku.probability_print(board_array, prediction.prob)

# %%

def interactive_train(net):
    pass

# %%
def detail(net, X):
    print(torch.mean(X))
    for i, layer in enumerate(net.shared):
        print(layer.__class__.__name__,'output shape: \t',X.shape)
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            print(f'shape: {layer.weight.shape}')
            print(f'weight:{torch.mean(layer.weight).item():.6f}, bias: {torch.mean(layer.bias).item():.6f}')
        X = layer(X)
        print(f'=> mean: {torch.mean(X).item():.6f}')
        with open(f'{layer.__class__.__name__}_{i}.log', 'w') as f:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        for u in range(X.shape[3]):
                            f.write(f'{X[i][j][k][u]:.3f} ')
                        f.write('\n')
                    f.write('\n')


# %%
if __name__ == '__main__':
    gomoku.init()
    gomoku.import_samples('data/5000ms.tr.dat')
    # if len(sys.argv) != 2:
    #     print(f'usage: {sys.argv[0]} [.dat file]')
    #     sys.exit(1)
    # if gomoku.import_samples(sys.argv[1]):
    #     sys.exit(2)
    test_size = gomoku.dataset_size() // 10
    print(f'dataset size: {gomoku.dataset_size()}, test size: {test_size}')

    batch_size = 256

    predictor = Predictor()

    X, policy, value = next(train_iter(batch_size))
    print(f'dataset output shape: {policy.shape}, {value.shape}')

# %%
    batch_size = 32
    num_epochs = 10
    lr = 0.005
    metrics = train(False, predictor, train_iter(batch_size), test_iter(batch_size), num_epochs, lr, try_mps())
    animator1 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], 
                             legend=['train value loss', 'test value loss'])
    animator2 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[2, 5], 
                             legend=['train policy loss', 'entropy'])
    for i, metric in enumerate(metrics):
        animator1.add(i + 1, (metric[0], metric[1]))
        animator2.add(i + 1, (metric[2], metric[3]))
    predictor.to(cpu())

    print(f'completed, loss: {metrics[-1][0]:.3f}, {metrics[-1][2]:.3f}, test loss {metrics[-1][1]:.3f}, entropy {metrics[-1][3]:.3f}')

    # predictor.save("model/predictor_min.params")
    # print('start load')
    # predictor.load("model/predictor_min.params")
    # input('press enter to continue')
    # predictor.to_ctype("model/predictor_min_tmp.mod")

    # ctype_net = predictor.to_ctype()
    # gomoku.predictor_save(ctypes.pointer(ctype_net), 'model/predictor_min.tmp.mod')

    # ctype_net = gomoku.predictor_network_t()
    # gomoku.predictor_load(ctypes.pointer(ctype_net), 'model/predictor_min.tmp.mod')

# %%
    # for i in range(gomoku.dataset_size()):
    #     sample=gomoku.find_sample(i)
    #     if sample.winner == 0:
    #         print(i)
    #         continue
    #     gomoku.print_sample(sample)
    #     test_sample(predictor, ctype_net, sample)
    #     input('press enter to continue')
# %%
