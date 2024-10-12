import lib.gomoku as gomoku
import ctypes
import signal
import sys
import torch
import torch.nn as nn
import numpy as np

nullptr = ctypes.POINTER(ctypes.c_int)()

gomoku.init()

in_channels = 3
out_channels1 = 32
out_channels2 = 64
kernel_size = 3
padding = 1
fc1_input_dim = out_channels2 * 15 * 15
fc1_output_dim = 128
fc2_output_dim = 15 * 15 + 1

# 定义卷积神经网络模型
class GomokuNN(nn.Module):
    def __init__(self):
        super(GomokuNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(fc1_input_dim, fc1_output_dim)
        self.fc2 = nn.Linear(fc1_output_dim, fc2_output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GomokuNNWrapper(ctypes.Structure):
    _fields_ = [
        ("conv1_weight", ctypes.c_float * (out_channels1 * in_channels * kernel_size * kernel_size)),
        ("conv1_bias", ctypes.c_float * out_channels1),
        ("conv2_weight", ctypes.c_float * (out_channels2 * out_channels1 * kernel_size * kernel_size)),
        ("conv2_bias", ctypes.c_float * out_channels2),
        ("fc1_weight", ctypes.c_float * (fc1_output_dim * fc1_input_dim)),
        ("fc1_bias", ctypes.c_float * fc1_output_dim),
        ("fc2_weight", ctypes.c_float * (fc2_output_dim * fc1_output_dim)),
        ("fc2_bias", ctypes.c_float * fc2_output_dim),
    ]

# 对应的C结构体
"""
typedef struct {
    float conv1_weight[out_channels1 * in_channels * kernel_size * kernel_size];
    float conv1_bias[out_channels1];
    float conv2_weight[out_channels2 * out_channels1 * kernel_size * kernel_size];
    float conv2_bias[out_channels2];
    float fc1_weight[fc1_output_dim * fc1_input_dim];
    float fc1_bias[fc1_output_dim];
    float fc2_weight[fc2_output_dim * fc1_output_dim];
    float fc2_bias[fc2_output_dim];
} neural_network_t;
"""

def export_model_to_ctypes(model):
    wrapper = gomoku.neural_network_t()
    np.copyto(np.ctypeslib.as_array(wrapper.conv1_weight), model.conv1.weight.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.conv1_bias), model.conv1.bias.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.conv2_weight), model.conv2.weight.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.conv2_bias), model.conv2.bias.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.fc1_weight), model.fc1.weight.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.fc1_bias), model.fc1.bias.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.fc2_weight), model.fc2.weight.detach().numpy().flatten())
    np.copyto(np.ctypeslib.as_array(wrapper.fc2_bias), model.fc2.bias.detach().numpy().flatten())
    return wrapper

# 预测函数
def predict(model, nn_input):
    # 将输入转换为 numpy 数组
    input_array = np.array(nn_input.input, dtype=np.float32).reshape((1, 3, 15, 15))
    
    # 将 numpy 数组转换为 PyTorch 张量
    input_tensor = torch.tensor(input_array)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor).numpy()
    
    nn_output = gomoku.nn_output_t()
    nn_output.prob = (output[0, :-1].reshape((15, 15))).astype(np.float64)
    nn_output.value = output[0, -1].astype(np.float64)
    
    return nn_output

"""
# 示例用法
if __name__ == "__main__":
    # 创建模型
    model = GomokuNN()
    
    # 创建示例输入
    nn_input = gomoku.nn_input_t()
    nn_input.input = np.random.rand(3, 15, 15).astype(np.float64)
    
    # 进行预测
    nn_output = predict(model, nn_input)
    
    # 打印预测结果
    print("Probabilities:", nn_output.prob)
    print("Value:", nn_output.value)
"""

def start_game(player1, player2, first_player): 
    game = gomoku.new_game(first_player)
    gomoku.game_print(game)
    while True:
        id = game.cur_player
        pos = gomoku.move(player1 if id == 1 else player2, nullptr, game)
        gomoku.game_add_step(ctypes.pointer(game), pos)
        gomoku.game_print(game)
        if gomoku.check_draw(game.board):
            print('draw')
            return game
        if gomoku.check(game.board, pos):
            print('player %d win' % id)
            return game

def signal_handler(sig, frame):
    print('exited trainer')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
start_game(gomoku.MCTS, gomoku.MCTS, 1)
