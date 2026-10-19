"""CUDA device selection without importing the model or sparse backbone."""

import torch


def resolve_cuda_device(specification):
    device = torch.device(specification)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full sparse backbone requires CUDA. Run in the server TreeLearn environment.')
    # Older torch CUDA memory APIs reject torch.device('cuda') without an
    # index. Resolve the logical current device, respecting CUDA_VISIBLE_DEVICES.
    index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(index)
    return torch.device('cuda', index)
