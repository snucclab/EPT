import math
from typing import Union

import torch
from torch import nn

from page.const import PAD_ID


class PositionalEncoding(nn.Module):
    """
    Positional encoding that extends trigonometric embedding proposed in 'Attention is all you need'
    """

    def __init__(self, embedding_dim):
        """
        Instantiate positional encoding instance.

        :param int embedding_dim:
            Dimension of embedding vector
        """

        super().__init__()
        #: Dimension of embedding vector
        self.embedding_dim = embedding_dim

        # The output will be c_p * cos(a_p * t + b_p) + d_p * sin(a_p * t + b_p), where t=index and p = 1...embed_dim
        # From "Attention is all you need" paper.
        # Here, b_p = 0 and a_2p = a_{2p+1} = 1 / 10000^{2p/embed_dim}.
        # Thus, we need to define a_p only.
        div_term = (torch.arange(0, embedding_dim) // 2) * 2
        div_term = torch.exp(div_term.float() * (-math.log(10000.0) / embedding_dim))
        # Note: c_p = 1 if p is odd, 0 otherwise and d_p = 1 if p is even, 0 otherwise
        multiplier = torch.zeros(2, embedding_dim, dtype=torch.float)
        multiplier[0, 1::2] = 1.0  # Only use cosine for odd indices
        multiplier[1, 0::2] = 1.0  # Only use sine for even indices

        # Fix a_p, c_p, d_p values.
        self.register_buffer('_div_term', div_term)
        self.register_buffer('multiplier', multiplier)

    @property
    def device(self) -> torch.device:
        """
        Get the device where weights are currently put.
        :rtype: torch.device
        :return: Device instance
        """
        return self._div_term.device

    def before_trigonometric(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute a_p * t + b_p for each index t.
        :param torch.Tensor indices: A Long tensor to compute indices.
        :rtype: torch.Tensor
        :return: Tensor whose values are a_p * t + b_p for each (t, p) entry.
        """
        indices = indices.float()

        # Compute a_p * t.
        return indices * self._div_term

    def forward(self, index_or_range: Union[torch.Tensor, int, range], ignored_index=PAD_ID) -> torch.Tensor:
        """
        Compute positional encoding. If this encoding is not learnable, the result cannot have any gradient vector.

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        # we don't need to compute gradients.
        with torch.no_grad():
            return self._forward(index_or_range, ignored_index)

    def _forward(self, index_or_range: Union[torch.Tensor, int, range], ignored_index=PAD_ID) -> torch.Tensor:
        """
        Compute positional encoding

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        if type(index_or_range) is int:
            # Build Long Tensor of [0, ..., index-1]
            indices = torch.arange(0, index_or_range)
        elif type(index_or_range) is range:
            # Build Long Tensor of [range]
            indices = torch.as_tensor(list(index_or_range))
        else:
            indices = index_or_range

        # Unsqueeze the last dimension to pass the linear layer.
        indices = indices.unsqueeze(-1)

        # Send indices to device that currently using.
        indices = indices.to(self.device)

        # Now indices will have shape [*, 1], we can apply the linear layer, a_p * t + b_p.
        phase = self.before_trigonometric(indices)

        # Phase has shape [*, E]. Apply cosine and sine function on the phase.
        cos_value = phase.cos()
        sin_value = phase.sin()

        # Retrieve c_p and d_p vectors. These have shape [E].
        cos_multiplier = self.multiplier[0]
        sin_multiplier = self.multiplier[1]

        # To multiply c_p and d_p on [*, E], unsqueeze c_p and d_p to fit [*].
        # Make the dimension of c_p the same
        result_shape = [1] * (phase.dim() - 1) + [-1]
        cos_multiplier = cos_multiplier.view(*result_shape)
        sin_multiplier = sin_multiplier.view(*result_shape)

        # Compute c_p * cos(phase) + d_p * sin(phase). Shape will be [*, E].
        result = cos_value * cos_multiplier + sin_value * sin_multiplier

        # Fill ignored indices as zero.
        ignored_indices = (indices == ignored_index)
        if ignored_indices.any():
            result.masked_fill_(ignored_indices, 0.0)

        # Return value. Shape [*, E]
        return result.contiguous()


__all__ = ['PositionalEncoding']
