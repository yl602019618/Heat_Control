import torch
import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def clip_tensor(inp,min,max):
    result = (inp >= min)*inp + (inp<min)*min
    result = (result <= max)*result + (result>max)*max
    return result


def nptotorch(x):
    if isinstance(x,torch.Tensor):
        return x
    elif isinstance(x,np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    else: torch.tensor(x)

