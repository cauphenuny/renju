import torch
import ctypes

def to_ctype(item):
    if isinstance(item, torch.Tensor):
        item = item.flatten()
        return (ctypes.c_float * item.shape[0])(*item)
    if isinstance(item, torch.nn.Linear):
        return to_ctype(item.weight), to_ctype(item.bias)
    if isinstance(item, torch.nn.Conv2d):
        return to_ctype(item.weight), to_ctype(item.bias)
    
    raise ValueError(f"cannot convert {type(item)} to ctype")
