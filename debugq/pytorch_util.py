import numpy as np
import torch

USE_GPU = torch.cuda.is_available()
CUDA_DEVICE = torch.device('cuda')
CPU_DEVICE = torch.device('cpu')

def set_gpu(enable=True):
    global USE_GPU
    USE_GPU = enable

def get_device():
    if USE_GPU:
        return CUDA_DEVICE
    else:
        return CPU_DEVICE

def tensor(array, pin=False, dtype=torch.float32):
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array
    device = get_device()
    var = torch.tensor(array, dtype=dtype, device=device)
    if pin:
        var.pin_memory()
    return var

def all_tensor(arrays, **kwargs):
    return [tensor(arr, **kwargs) for arr in arrays]

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if USE_GPU:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def initialize_network(network):
    if USE_GPU:
        network.cuda()

def logsumexp(input, dim, keepdim=False, alpha=1.0):
    if alpha == 1.0:
        return torch.logsumexp(input, dim, keepdim=keepdim)
    else:
        return alpha * torch.logsumexp( input/alpha, dim, keepdim=keepdim)

def one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=get_device()).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)