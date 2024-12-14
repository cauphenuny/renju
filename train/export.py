import torch
import ctypes

def to_ctype(item):
    if isinstance(item, torch.Tensor):
        item = item.flatten()
        return (ctypes.c_float * item.shape[0])(*item)
    if hasattr(item, 'weight') and hasattr(item, 'bias'):
        return to_ctype(item.weight), to_ctype(item.bias)
    
    raise TypeError(f"cannot convert {type(item)} to ctype")
