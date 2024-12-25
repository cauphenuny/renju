# %%
import os
import time
import numpy
from lib import librenju as renju
from collections import OrderedDict
import torch
import torch.nn as nn
import ctypes
from layer import Linear, Conv2d, ResidualBlock, to_renju_tensor, init_renju_tensor
from layer import try_mps, try_cuda, cpu

class Predictor(nn.Module):
    def __init__(self, model_name=None, device=cpu()):
        super().__init__()
        self.max_channel = renju.MAX_CHANNEL
        self.shared = nn.Sequential(OrderedDict([
            ('res1', ResidualBlock(4, 32)), 
            ('res2', ResidualBlock(32, 256)), 
            ('dropout', nn.Dropout(0.3)),
            ('res3', ResidualBlock(256, self.max_channel)),
        ]))
        self.policy = nn.Sequential(OrderedDict([
            ('res', ResidualBlock(self.max_channel, 16)),
            ('flatten', nn.Flatten()),
            ('linear', Linear(16 * 15 * 15, 225, "softmax")),
        ]))
        self.value = nn.Sequential(OrderedDict([
            ('conv', Conv2d(self.max_channel, 16, kernel_size=3, padding=1, activate="relu")), 
            ('flatten', nn.Flatten()),
            ('linear1', Linear(16 * 15 * 15, 256, "relu")),
            ('linear2', Linear(256, 3, "softmax")),
        ]))
        self.model_name = model_name
        self.device = device
        super().to(self.device)
        if model_name != None:
            self.load(model_name)

    def forward(self, x):
        # shared_output = x.to(self.device)
        # for sh in self.shared:
        #     print(sh)
        #     shared_output = sh(shared_output)
        #     data = shared_output.cpu().flatten().detach().numpy()
        #     print(f"res result mean: {numpy.mean(data):.7f}")

        shared_output = self.shared(x.to(self.device))

        policy_output = self.policy(shared_output)
        value_output = self.value(shared_output)
        return policy_output, value_output

    def save(self, basename, device=try_mps()):
        dev = self.device
        self.to(device)
        name = f'{basename}.v{renju.NETWORK_VERSION}.{self.max_channel}ch.{device}.params'
        torch.save(self.state_dict(), name)
        print(f'saved params to {name}')
        self.to(dev)

    def load(self, basename, device=try_mps()):
        dev = self.device
        self.to(device)
        name = f'{basename}.v{renju.NETWORK_VERSION}.{self.max_channel}ch.{device}.params'
        print(f'load params from {name}')
        self.load_state_dict(torch.load(name, weights_only=False))
        self.to(dev)

    def to(self, device):
        self.device = device
        super().to(device)

    def to_ctype(self):
        dev = self.device
        self.to(cpu())
        layers = [
            'shared.res1', 'shared.res2', 'shared.res3',
            'value.conv', 'value.linear1', 'value.linear2',
            'policy.res', 'policy.linear',
        ]
        net = renju.network_t()
        for i, layer in enumerate(layers):
            exec(f"net.{layer} = self.{layer}.to_ctype()")
            print(f'converted {layer}, {i+1}/{len(layers)}')
        self.to(dev)
        return net

    def export_ctype(self, name = None):
        if name == None:
            name = self.model_name
        if name == None:
            name = input('input model name: ')
        print(f'exporting model "{name}"')
        ctype_net = self.to_ctype()
        renju.save_network(ctypes.pointer(ctype_net), name)

# %%
def test_predictor(pred, ctype_pred):
    print(pred.shared.res1.conv1.conv.bias.mean())
    input_tensor = torch.rand(1, 4, 15, 15)
    start = time.time()
    pred.to(try_mps())
    policy_output, value_output = pred(input_tensor)
    end = time.time()
    print(f'forward time: {(end - start) * 1000:.2f}ms')
    print(f'policy: {numpy.round(torch.exp(policy_output).cpu().flatten().detach().numpy(), 2)}')
    print(f'value: {numpy.round(torch.exp(value_output).cpu().flatten().detach().numpy(), 2)}')
    
    # renju.print_tensor(ctype_pred.shared.res1.conv3x3_1.weight)

    renju_input_tensor, storage1 = to_renju_tensor(input_tensor, start=1)
    # renju.print_tensor(renju_input_tensor)
    renju_policy_output = init_renju_tensor(list(policy_output.shape))
    renju_value_output = init_renju_tensor(list(value_output.shape))
    ptr = ctypes.pointer
    renju.forward(ptr(ctype_pred), ptr(renju_input_tensor), ptr(renju_policy_output), ptr(renju_value_output))
    print("renju forward finished")
    renju.print_tensor(ptr(renju_policy_output))
    renju.print_tensor(ptr(renju_value_output))

# %%
if __name__ == "__main__":
    pred = Predictor()
    # pred.load("model/static")
    ctype_pred = pred.to_ctype()
    # renju.network_save(ctypes.pointer(ctype_pred), "model/static")
# %%
    test_predictor(pred, ctype_pred)

# %%
