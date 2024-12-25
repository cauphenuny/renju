import ctypes
import sys
import lib.librenju as rj
from predictor import Predictor
from trainer import test_sample
import time
import random

rj.init()
dataset_file = sys.argv[1]
model_name = sys.argv[2]

dataset = rj.dataset_t()
ptr = ctypes.pointer(dataset)
rj.load_dataset(ptr, dataset_file)

net = Predictor()
net.load(model_name)
# ctype_net = net.to_ctype()

random.seed(time.time())

while True:
    sample = rj.random_sample(ptr)
    test_sample(net, None, sample)
    input("press enter to continue... ")
