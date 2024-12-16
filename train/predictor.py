# %%
#!/usr/bin/env python3

from IPython import display
from colorama import Fore
from lib import libgomoku as gomoku
from d2l import torch as d2l
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes
import sys
from dataset import GomokuDataset, to_tensor
from export import to_ctype
import random

def cpu():
    return torch.device("cpu")

def try_cuda():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def try_mps():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return try_cuda() 

def evaluate_accuracy(net, device, iter, loss1, loss2):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X, y2, y1 in iter:
            X, y2, y1 = X.to(device), y2.to(device), y1.to(device)
            y2_hat, y1_hat = net(X)
            l1, l2 = loss1(y1_hat, y1), loss2(y2_hat, y2)
            metric.add(X.shape[0], X.shape[0] * l1, X.shape[0] * l2)
    return metric[1] / metric[0], metric[2] / metric[0]

def evaluate_accuracy(net, device, iter):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X, y2, y1 in iter:
            X, y2, y1 = X.to(device), y2.to(device), y1.to(device)
            y2_hat, y1_hat = net(X)
            metric.add(X.shape[0], 
                       torch.sum(torch.argmax(y1_hat, dim=1) == torch.argmax(y1, dim=1)).item(), 
                       torch.sum(torch.argmax(y2_hat, dim=1) == torch.argmax(y2, dim=1)).item())
    return metric[1] / metric[0], metric[2] / metric[0]

class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.content = layer
        self.weight = layer.weight
        self.bias = layer.bias
    
    def forward(self, X):
        return X + self.content(X)

# %%
class Predictor(nn.Module):
    def __init__(self, max_channel=gomoku.MAX_CHANNEL, model_name=None, device=try_mps()):
        super().__init__()
        self.shared = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 32, kernel_size=5, padding=2)), ('relu1', nn.ReLU()), 
            ('dropout1', nn.Dropout()),
            ('conv2', nn.Conv2d(32, 64, kernel_size=5, padding=2)), ('relu2', nn.ReLU()), 
            ('conv3', nn.Conv2d(64, max_channel, kernel_size=3, padding=1)), ('relu3', nn.ReLU()), 
            ('dropout2', nn.Dropout()),
        ]))
        self.value = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(max_channel, 4, kernel_size=3, padding=1)), ('relu1', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(4 * 15 * 15, 3)), 
            ('logsoftmax', nn.LogSoftmax(dim=1)),
        ]))
        self.policy = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(max_channel, 32, kernel_size=3, padding=1)), ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 1, kernel_size=3, padding=1)), ('relu2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', Residual(nn.Linear(15 * 15, 225))), 
            ('logsoftmax', nn.LogSoftmax(dim=1)),
        ]))
        self.max_channel = max_channel
        self.to(device)
        self.model_name = model_name
        if model_name != None:
            self.load(model_name)

    def forward(self, x):
        shared_output = self.shared(x)
        policy_output = self.policy(shared_output)
        value_output = self.value(shared_output)
        return policy_output, value_output

    def save(self, basename):
        name = f'{basename}.v{gomoku.NETWORK_VERSION}.{self.max_channel}ch.params'
        torch.save(self.state_dict(), name)
        print(f'saved params to {name}')

    def load(self, basename):
        name = f'{basename}.v{gomoku.NETWORK_VERSION}.{self.max_channel}ch.params'
        self.to(try_mps())
        print(f'load params from {name}')
        self.load_state_dict(torch.load(name))

    def to_ctype(self):
        layers = [
            'shared.conv1', 'shared.conv2', 'shared.conv3',
            'value.conv', 'value.linear',
            'policy.conv1', 'policy.conv2', 'policy.linear',
        ]
        net = gomoku.network_t()
        for i, layer in enumerate(layers):
            exec(f"net.{layer}.weight, net.{layer}.bias = to_ctype(self.{layer})")
            print(f'converted {layer}, {i+1}/{len(layers)}')
        return net

    def export_ctype(self, name = None):
        if name == None:
            name = self.model_name
        if name == None:
            name = input('input model name: ')
        print(f'exporting model "{name}"')
        ctype_net = self.to_ctype()
        gomoku.save_network(ctypes.pointer(ctype_net), name)

# %%
def calculate_entropy(tensor):
    return -torch.sum(torch.exp(tensor) * tensor)

def print_sample(sample):
    gomoku.print_sample(sample)
    X, prob, eval = to_tensor(sample)

    print(f'eval: {eval}')
    print(f'entropy: {calculate_entropy(prob)}')
    print(f'prob detail: {prob}')

# %%
def cross_entropy_loss(log_y_hat, y):
    return -torch.mean(torch.sum(y * log_y_hat, dim=1))

def train(net, train_iter, test_iter, num_epochs, lr, device):
    print(f'training on {device}, lr: {lr}, num_epochs: {num_epochs}')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    timer, num_batches = d2l.Timer(), len(train_iter)
    print(f'num_batches: {num_batches}')
    # mse_loss = nn.MSELoss()
    metric_record = []
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(5)
        net.train()
        cnt = 0
        acc_count = 0
        for i, (X, policy, value) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, policy, value = X.to(device), policy.to(device), value.to(device)
            policy_hat, value_hat = net(X)
            value_loss = cross_entropy_loss(value_hat, value)
            policy_loss = cross_entropy_loss(policy_hat, policy) 
            l = value_loss + policy_loss
            l2_lambda = 0.001
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in net.parameters():
                l2_reg = l2_reg + torch.norm(param)
            l += l2_lambda * l2_reg
            l.backward()
            optimizer.step()
            acc_count += torch.sum(torch.argmax(value_hat, dim=1) == torch.argmax(value, dim=1)).item()
            with torch.no_grad():
                metric.add(X.shape[0], 
                           value_loss * X.shape[0],
                           policy_loss * X.shape[0],
                           sum(calculate_entropy(policy_hat[i]) for i in range(X.shape[0])), 
                           sum(calculate_entropy(value_hat[i]) for i in range(X.shape[0])))
            timer.stop()
            cnt += 1
        # print(cnt)
        train_loss1 = metric[1] / metric[0]
        train_loss2 = metric[2] / metric[0]
        train_acc = acc_count / metric[0]
        entropy = metric[3] / metric[0]
        value_entropy = metric[4] / metric[0]
        if test_iter != None:
            test_acc1, test_acc2 = evaluate_accuracy(net, device, test_iter)
        else:
            test_acc1, test_acc2 = 0, 0
        metric_record.append([train_loss1, train_loss2, test_acc1, test_acc2, entropy])
        print(f'epoch {epoch + 1}, loss {train_loss1:.3f}, {train_loss2:.3f}, acc {train_acc:.3f}, test acc {test_acc1:.3f}, {test_acc2:.3f}, val entropy {value_entropy:.3f}, entropy {entropy:.3f}')
    print(f'value loss {train_loss1:.3f}, policy loss {train_loss2:.3f}, test acc {test_acc1:.3f}, {test_acc2:.3f}, entropy {entropy:.3f}')
    print(f'{metric[0] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    return metric_record

# %%
def test_sample(net, ctype_net, sample):
    gomoku.print_sample(sample)

    X, prob, eval = to_tensor(sample)
    X = X.reshape(1, 4, 15, 15).to(try_mps())
    net.to(try_mps())
    net.eval()
    log_prob_hat, eval_hat = net(X)
    prob_hat = torch.exp(log_prob_hat)

    print("probability:")
    # print(prob_hat)
    board_array = gomoku.board_t()
    prob_array = gomoku.fboard_t()
    for x in range(15):
        for y in range(15):
            if sample.input.p1_pieces[x][y] != 0:
                board_array[x][y] = 1
            elif sample.input.p2_pieces[x][y] != 0:
                board_array[x][y] = 2
            else:
                board_array[x][y] = 0
            prob_array[x][y] = prob_hat[0][x * 15 + y].item()
    gomoku.print_prob(board_array, prob_array)
    print(f'entropy: {calculate_entropy(log_prob_hat[0]).item():.3f}')
    print(f'eval: {eval_hat[0][0].item():.3f}')

    if ctype_net != None:
        cur_id = sample.input.current_player
        if cur_id == -1:
            cur_id = 2
        print('call gomoku.predict...')
        prediction = gomoku.predict(ctypes.pointer(ctype_net), 
                                    board_array, 
                                    sample.input.last_move, 
                                    1, 
                                    cur_id)
        gomoku.print_prediction(prediction)

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

class CombinedIterator:
    def __init__(self, iterators):
        self.iterators = iterators
        self.available = list(range(len(iterators)))

    def __len__(self):
        return sum(len(it) for it in self.iterators)

    def __iter__(self):
        self.available = list(range(len(self.iterators)))
        return self

    def __next__(self):
        if not self.available:
            raise StopIteration
        index = random.choice(self.available)
        try:
            return next(self.iterators[index])
        except StopIteration:
            self.available.remove(index)
            return self.__next__()

# %%
if __name__ == '__main__':
    gomoku.init()
    if len(sys.argv) != 3:
        train_files = ["data/3000ms.dat", "data/5000ms.dat"]
        test_file = "data/3000ms-raw.dat"
    else:
        train_files = [sys.argv[1]]
        test_file = sys.argv[2]
    train_iters = lambda batch_size: [GomokuDataset(file=train_file, batch_size=batch_size, device=try_mps()) for train_file in train_files]
    train_iter = lambda batch_size: CombinedIterator(train_iters(batch_size))
    test_iter = lambda batch_size: GomokuDataset(file=test_file, batch_size=batch_size, device=try_mps())

    batch_size = 64

    network = Predictor(128)
    X, policy, value = next(train_iter(batch_size))
    print(f'dataset output shape: {policy.shape}, {value.shape}')

    policy_hat, value_hat = network(X)
    print(f'network output shape: {policy_hat.shape}, {value_hat.shape}')
    network.load("model/static")

# %%
    num_epochs = 10
    lr = 0.005
    batch_size = 32
    metrics = train(network, train_iter(batch_size), test_iter(batch_size), num_epochs, lr, try_mps())
    animator1 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.5], 
                             legend=['train value loss', 'test value loss'])
    animator2 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[2, 5], 
                             legend=['train policy loss', 'test policy loss', 'entropy'])
    for i, metric in enumerate(metrics):
        animator1.add(i + 1, (metric[0], metric[1]))
        animator2.add(i + 1, (metric[2], metric[3], metric[4]))
    network.to(cpu())

    print(f'completed, loss: {metrics[-1][0]:.3f}, {metrics[-1][2]:.3f}, test loss {metrics[-1][1]:.3f}, entropy {metrics[-1][3]:.3f}')
# %%
    name = input('input model name: ')
    if name != "":
        name = f'model/{name}'
        network.save(name)
        network.export_ctype(name)

    # network.save("model/network_min.params")
    # print('start load')
    # network.load("model/network_min.params")
    # input('press enter to continue')
    # network.to_ctype("model/network_min_tmp.mod")

    # ctype_net = network.to_ctype()
    # gomoku.save_network(ctypes.pointer(ctype_net), 'model/network_min.tmp.mod')

    # ctype_net = gomoku.network_network_t()
    # gomoku.network_load(ctypes.pointer(ctype_net), 'model/network_min.tmp.mod')

    # for i in range(gomoku.dataset_size()):
    #     sample=gomoku.find_sample(i)
    #     if sample.winner == 0:
    #         print(i)
    #         continue
    #     gomoku.print_sample(sample)
    #     test_sample(network, ctype_net, sample)
    #     input('press enter to continue')
# %%