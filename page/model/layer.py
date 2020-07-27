import torch
from torch import nn

from page.const import NEG_INF


class AveragePooling(nn.Module):
    """
    Layer class for computing mean of a sequence
    """

    def __init__(self, dim: int = -1, keepdim: bool = False):
        """
        Layer class for computing mean of a sequence

        :param int dim: Dimension to be averaged. -1 by default.
        :param bool keepdim: True if you want to keep averaged dimensions. False by default.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Do average pooling over a sequence

        :param torch.Tensor tensor: FloatTensor to be averaged
        :rtype: torch.FloatTensor
        :return: Averaged result
        """
        return tensor.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        # Extra representation for repr()
        return 'dim={dim}, keepdim={keepdim}'.format(**self.__dict__)


class Squeeze(nn.Module):
    """
    Layer class for squeezing a dimension
    """

    def __init__(self, dim: int = -1):
        """
        Layer class for squeezing a dimension

        :param int dim: Dimension to be squeezed, -1 by default.
        """
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Do squeezing

        :param torch.Tensor tensor: FloatTensor to be squeezed
        :rtype: torch.FloatTensor
        :return: Squeezed result
        """
        return tensor.squeeze(dim=self.dim)

    def extra_repr(self):
        # Extra representation for repr()
        return 'dim={dim}'.format(**self.__dict__)


class LogSoftmax(nn.LogSoftmax):
    """
    LogSoftmax layer that can handle infinity values.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute log(softmax(tensor))

        :param torch.Tensor tensor: FloatTensor whose log-softmax value will be computed
        :rtype: torch.FloatTensor
        :return: LogSoftmax result.
        """
        # Find maximum values
        max_t = tensor.max(dim=self.dim, keepdim=True).values
        # Reset maximum as zero if it is a finite value.
        tensor = (tensor - max_t.masked_fill(~torch.isfinite(max_t), 0.0))

        # If a row's elements are all infinity, set the row as zeros to avoid NaN.
        all_inf_mask = torch.isinf(tensor).all(dim=self.dim, keepdim=True)
        if all_inf_mask.any().item():
            tensor = tensor.masked_fill(all_inf_mask, 0.0)

        # Forward nn.LogSoftmax.
        return super().forward(tensor)


class InfinityAwareLinear(nn.Linear):
    """
    Linear layer that can handle infinity values.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute linear(tensor)

        :param torch.Tensor tensor: FloatTensor whose linear transformation will be computed
        :rtype: torch.FloatTensor
        :return: Linearly transformed result.
        """
        is_inf = ~torch.isfinite(tensor)
        tensor_masked = tensor.masked_fill(is_inf, 0.0)

        output = super().forward(tensor_masked)
        return output.masked_fill(is_inf.any(dim=-1, keepdim=True), NEG_INF)


__all__ = ['AveragePooling', 'Squeeze', 'LogSoftmax', 'InfinityAwareLinear']
