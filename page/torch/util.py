import torch
from torch import Tensor


def required_space_param(p: Tensor) -> float:
    """
    Compute required space to store parameters

    :param Tensor p: Tensor whose size will be computed
    :rtype: float
    :return: Size of the tensor in terms of bytes.
    """
    dtype = p.dtype
    numel = p.numel()

    if dtype == torch.bool:  # 1-bit
        return numel / 8.0
    elif dtype in [torch.uint8, torch.int8]:  # 8-bit, 1-byte
        return numel
    elif dtype in [torch.float16, torch.int16]:  # 16-bit, 2-byte
        return numel * 2.0
    elif dtype in [torch.float32, torch.int32]:  # 32-bit, 4-byte
        return numel * 4.0
    else:  # 64-bit, 8-byte
        return numel * 8.0


def get_available_device_count(default: int = 1) -> int:
    """
    Get the count of available GPU/CPU devices

    :param int default: Default value when GPU is not available.
    :rtype: int
    :return: The number of GPUs if any GPU is available, otherwise the default value.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return default


__all__ = ['required_space_param', 'get_available_device_count']
