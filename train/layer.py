import numpy
import torch
from torch import nn
from torch.nn import functional as F
from export import to_ctype
from lib import librenju as renju
import ctypes

nullptr = ctypes.POINTER(ctypes.c_void_p)()
ptr = ctypes.pointer

def to_renju_tensor(tensor):
    if len(tensor.shape) == 1:
        renju_tensor = renju.tensor1d_new(tensor.shape[0])
    elif len(tensor.shape) == 2:
        renju_tensor = renju.tensor2d_new(*list(tensor.shape))
    elif len(tensor.shape) == 3:
        renju_tensor = renju.tensor3d_new(*list(tensor.shape))
    elif len(tensor.shape) == 4:
        renju_tensor = renju.tensor4d_new(*list(tensor.shape))
    else:
        raise ValueError(f"unsupported tensor shape {tensor.shape}")
    content = to_ctype(tensor)
    renju_tensor.data = ctypes.cast(content, ctypes.POINTER(ctypes.c_float))
    return renju_tensor, content

def init_renju_tensor(shape):
    renju_tensor = renju.tensor_new(*shape, -1)
    return renju_tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activate):
        super().__init__()
        self.ifc = in_features
        self.ofc = out_features
        self.linear = nn.Linear(in_features, out_features)
        if activate is None:
            self.activate = None
            self.c_activate = nullptr
        elif activate == "relu":
            self.activate = F.relu
            self.c_activate = renju.relu
        elif activate == "softmax":
            self.activate = nn.LogSoftmax(dim=1)
            self.c_activate = renju.softmax
        elif activate == "tanh":
            self.activate = F.tanh
            self.c_activate = renju.tanh_
        else:
            raise ValueError(f"unknown activation function {activate}")
        self.weight_ctype = None
        self.bias_ctype = None

    def to_ctype(self):
        linear_layer = renju.linear_layer_t()
        linear_layer.param = renju.linear_params_t(self.ifc, self.ofc, 
                                                   ctypes.cast(self.c_activate, renju.activate_func_t))
        linear_layer.weight, self.weight_ctype = to_renju_tensor(self.linear.weight)
        linear_layer.bias, self.bias_ctype = to_renju_tensor(self.linear.bias)
        return linear_layer

    def forward(self, X):
        if self.activate:
            return self.activate(self.linear(X))
        else:
            return self.linear(X)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate):
        super().__init__()
        self.ich = in_channels
        self.och = out_channels
        self.ksize = kernel_size
        self.pd = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if activate is None:
            self.activate = None
            self.c_activate = nullptr
        elif activate == "relu":
            self.activate = F.relu
            self.c_activate = renju.relu
        elif activate == "softmax":
            self.activate = F.log_softmax
            self.c_activate = renju.softmax
        elif activate == "tanh":
            self.activate = F.tanh
            self.c_activate = renju.tanh_
        else:
            raise ValueError(f"unknown activation function {activate}")
        self.weight_ctype = None
        self.bias_ctype = None

    def to_ctype(self):
        conv_layer = renju.conv_layer_t()
        conv_layer.param = renju.conv_params_t(self.ich, self.och, 
                                               self.ksize, self.pd, 
                                               ctypes.cast(self.c_activate, renju.activate_func_t))
        conv_layer.weight, self.weight_ctype = to_renju_tensor(self.conv.weight)
        conv_layer.bias, self.bias_ctype = to_renju_tensor(self.conv.bias)
        return conv_layer

    def forward(self, X):
        if self.activate:
            return self.activate(self.conv(X))
        else:
            return self.conv(X)

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.ich = input_channel
        self.och = output_channel
        self.conv1 = Conv2d(input_channel, output_channel, kernel_size=3, padding=1, activate="relu")
        self.conv2 = Conv2d(output_channel, output_channel, kernel_size=3, padding=1, activate=None)
        if input_channel != output_channel:
            self.conv3 = Conv2d(input_channel, output_channel, kernel_size=1, padding=0, activate=None)
        else:
            self.conv3 = None

    def to_ctype(self):
        res_block = renju.residual_block_t()
        res_block.param = renju.residual_block_param_t(self.ich, self.och)
        res_block.conv3x3_1 = self.conv1.to_ctype()
        res_block.conv3x3_2 = self.conv2.to_ctype()
        if self.conv3:
            res_block.conv1x1 = self.conv3.to_ctype()
        return res_block

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        if self.conv3:
            Y2 = self.conv3(X)
        else:
            Y2 = X
        return F.relu(Y + Y2)

# %%
def test_residual_block():
    block = ResidualBlock(3, 4)
    print("residual block initialized")
    print(block.conv1.conv.weight.shape)
    print(block.conv1.conv.bias.shape)
    with torch.no_grad():
        block.conv1.conv.weight.uniform_(-0.1, 0.1)
        block.conv1.conv.bias.uniform_(-0.1, 0.1)
        block.conv2.conv.weight.uniform_(-0.1, 0.1)
        block.conv2.conv.bias.uniform_(-0.1, 0.1)
        if block.conv3:
            block.conv3.conv.weight.uniform_(-0.1, 0.1)
            block.conv3.conv.bias.uniform_(-0.1, 0.1)

    print(block.conv1.conv.weight)
    print(block.conv1.conv.bias)
    print(block.conv2.conv.weight)
    print(block.conv2.conv.bias)
    if block.conv3:
        print(block.conv3.conv.weight)
        print(block.conv3.conv.bias)

    input_tensor = torch.randn(3, 4, 4)
    print(f"input initialized: {input_tensor}")

    output_tensor = block(input_tensor)
    print(f'output: {numpy.round(output_tensor.flatten().detach().numpy(), 2)}')

    renju_block = block.to_ctype()
    renju_input_tensor, storage1 = to_renju_tensor(input_tensor)
    renju.print_tensor(renju_input_tensor)
    renju_output_tensor = init_renju_tensor(list(output_tensor.shape))
    renju.residual_block(ptr(renju_block), ptr(renju_input_tensor), ptr(renju_output_tensor))
    renju.print_tensor(ptr(renju_output_tensor))
    
def test_linear():
    linear = Linear(3, 4, "relu")
    print("linear layer initialized")
    print(linear.linear.weight.shape)
    print(linear.linear.bias.shape)
    with torch.no_grad():
        linear.linear.weight.uniform_(-0.1, 0.1)
        linear.linear.bias.uniform_(-0.1, 0.1)
    print(linear.linear.weight)
    print(linear.linear.bias)

    input_tensor = torch.randn(3)
    print(f"input initialized: {input_tensor}")

    output_tensor = linear(input_tensor)
    print(f'output: {numpy.round(output_tensor.detach().numpy(), 2)}')

    renju_linear = linear.to_ctype()
    renju_input_tensor, storage1 = to_renju_tensor(input_tensor)
    renju.print_tensor(renju_input_tensor)
    renju_output_tensor = init_renju_tensor(list(output_tensor.shape))
    renju.linear_layer(ptr(renju_linear), ptr(renju_input_tensor), ptr(renju_output_tensor))
    renju.print_tensor(ptr(renju_output_tensor))

def test_conv():
    conv = Conv2d(3, 4, 3, 1, "relu")
    print("conv layer initialized")
    print(conv.conv.weight.shape)
    print(conv.conv.bias.shape)
    with torch.no_grad():
        conv.conv.weight.uniform_(-0.1, 0.1)
        conv.conv.bias.uniform_(-0.1, 0.1)
    print(conv.conv.weight)
    print(conv.conv.bias)

    input_tensor = torch.randn(3, 4, 4)
    print(f"input initialized: {input_tensor}")

    output_tensor = conv(input_tensor)
    print(f'output: {numpy.round(output_tensor.flatten().detach().numpy(), 2)}')

    renju_conv = conv.to_ctype()
    renju_input_tensor, storage1 = to_renju_tensor(input_tensor)
    renju.print_tensor(renju_input_tensor)
    renju_output_tensor = init_renju_tensor(list(output_tensor.shape))
    renju.conv2d_layer(ptr(renju_conv), ptr(renju_input_tensor), ptr(renju_output_tensor), 0)
    renju.print_tensor(ptr(renju_output_tensor))

if __name__ == "__main__":
    while True:
        name = input("test name: ")
        exec(f"test_{name}()")
