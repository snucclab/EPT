from typing import Dict, Tuple

import torch
from torch import nn

from page.config import ModelConfig
from page.const import *


def gather_vectors(hidden: torch.Tensor, mask: torch.Tensor, max_len: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather hidden states of indicated positions.

    :param torch.Tensor hidden:
        Float Tensor of hidden states.
        Shape [B, S, H], where B = batch size, S = length of sequence, and H = hidden dimension
    :param torch.Tensor mask:
        Long Tensor which indicates number indices that we're interested in. Shape [B, S].
    :param int max_len:
        Expected maximum length of vectors per batch. 1 by default.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :return:
        Tuple of Tensors:
        - [0]:  Float Tensor of indicated hidden states.
                Shape [B, N, H], where N = max(number of interested positions, max_len)
        - [1]:  Bool Tensor of padded positions.
                Shape [B, N].
    """
    # Compute the maximum number of indicated positions in the text
    max_len = max(mask.max().item(), max_len)
    batch_size, seq_len, hidden_size = hidden.shape

    # Storage for gathering hidden states
    gathered = torch.zeros(batch_size, max_len, hidden_size, dtype=hidden.dtype, device=hidden.device)
    pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=hidden.device)

    # Average hidden states for tokens representing a number
    for row in range(batch_size):
        for i in range(max_len):
            indices = (mask[row] == i).nonzero().view(-1).tolist()

            if len(indices) > 0:
                begin = min(indices)
                end = max(indices) + 1

                # Copy masked positions. Take mean of number vectors.
                gathered[row, i] = hidden[row, begin:end].mean(dim=0)
                pad_mask[row, i] = False

    return gathered, pad_mask


class TextModel(nn.Module):
    """
    Model for encoding text.
    """

    def __init__(self, config: ModelConfig):
        """
        Initiate Text Model instance.

        :param ModelConfig config: Model configuration instance
        """
        super().__init__()
        self.model = config.load_encoder()

    def forward(self, max_numbers: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of encoding text.

        :param int max_numbers: Maximum number of numbers in the text. 1 by default.
        :keyword torch.Tensor text:
            Long Tensor representing text indices.
            Shape [B, S], where B = batch size, S = length of text
        :keyword torch.Tensor text_pad:
            Bool Tensor representing the position of padding in the text. Shape [B, S].
        :keyword torch.Tensor text_num:
            Bool Tensor representing the position of numbers in the text. Shape [B, S].
        :rtype: torch.Tensor
        :return:
            Dictionary of tensors.
            - [IN_TXT]: Encoded hidden states. Shape [B, S, H], where H = hidden dimension
            - [IN_TPAD]: text_pad.
            - [IN_TNUM]: Encoded hidden state of numbers. Shape [B, N, H], where N = the number of numbers in the text.
            - [IN_TNPAD]: Bool Tensor representing the position of padding in [IN_TNUM]. Shape [B, N].
        """
        text_pad = kwargs[IN_TPAD]

        # Encode text
        outputs = self.model(input_ids=kwargs[IN_TXT], attention_mask=(~text_pad).float())
        encoded = outputs[0]

        # Gather number vectors
        if IN_TNUM in kwargs:
            text_num, text_numpad = gather_vectors(encoded, kwargs[IN_TNUM], max_len=max_numbers)
        else:
            text_num = text_numpad = None

        return {
            IN_TXT: encoded,
            IN_TPAD: text_pad,
            IN_TNUM: text_num,
            IN_TNPAD: text_numpad
        }

    def save_pretrained(self, save_directory: str):
        """
        Save current state of Text Model.

        :param str save_directory: String that represents path to the directory where this will be saved.
        """
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, config: ModelConfig):
        """
        Restore the pretrained model using the specified configuration

        :param ModelConfig config: Configuration of a model to be restored
        :rtype: TextModel
        :return: A TextModel instance
        """
        return cls(config)


__all__ = ['TextModel']
