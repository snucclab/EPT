from typing import Dict, Union

import torch
from torch import nn
from torch.nn import functional as F

from page.const import PAD_ID
from .attention import MultiheadAttention, MultiheadAttentionWeights


def apply_module_dict(modules: nn.ModuleDict, encoded: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Predict next entry using given module and equation.

    :param nn.ModuleDict modules:
        Dictionary of modules to be applied. Modules will be applied with ascending order of keys.
        We expect three types of modules: nn.Linear, nn.LayerNorm and MultiheadAttention.
    :param torch.Tensor encoded:
        Float Tensor that represents encoded vectors.
        Shape [B, T, H], where B = batch size, T = length of equation, and H = hidden dimension.
    :keyword torch.Tensor key_value:
        Float Tensor that represents key and value vectors when computing attention.
        Shape [B, K, H], where K = length of keys
    :keyword torch.Tensor key_ignorance_mask:
        Bool Tensor whose True values at (b, k) make attention layer ignore k-th key on b-th item in the batch.
        Shape [B, K].
    :keyword attention_mask:
        Bool Tensor whose True values at (t, k) make attention layer ignore k-th key when computing t-th query.
        Shape [T, K].
    :rtype: torch.Tensor
    :return:
        Float Tensor that indicates the scores under given information. Shape will be [B, T, ?]
    """
    output = encoded
    keys = sorted(modules.keys())

    # Apply modules (ascending order of keys).
    for key in keys:
        layer = modules[key]
        if isinstance(layer, (MultiheadAttention, MultiheadAttentionWeights)):
            output = layer(query=output, **kwargs)
        else:
            output = layer(output)

    return output


def apply_across_dim(function, dim=1, shared_keys=None, **tensors) -> Dict[str, torch.Tensor]:
    """
    Apply a function repeatedly for each tensor slice through the given dimension.
    For example, we have tensor [B, X, S] and dim = 1, then we will concatenate the following matrices on dim=1.
    - function([:, 0, :])
    - function([:, 1, :])
    - ...
    - function([:, X-1, :]).

    :param function: Function to apply.
    :param int dim: Dimension through which we'll apply function. (1 by default)
    :param set shared_keys: Set of keys representing tensors to be shared. (None by default)
    :param torch.Tensor tensors: Keyword arguments of tensors to compute. Dimension should >= `dim`.
    :rtype: Dict[str, torch.Tensor]
    :return: Dictionary of tensors, whose keys are corresponding to the output of the function.
    """
    # Separate shared and non-shared tensors
    shared_arguments = {}
    repeat_targets = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or (shared_keys and key in shared_keys):
            shared_arguments[key] = tensor
        else:
            repeat_targets[key] = tensor

    # Check whether the size of the given dimension is the same across sliced_tensors.
    size = {key: tensor.shape[dim] for key, tensor in repeat_targets.items()}
    assert len(set(size.values())) == 1, 'Tensors does not have same size on dimension %s: We found %s' % (dim, size)

    # Since the sizes are the same, we will represent the size using the first entry.
    size = list(size.values())[0]

    # Dictionary for storing outputs
    output = {}

    for i in range(size):
        # Build kwargs for the function.
        kwargs = {key: tensor.select(dim=dim, index=i).contiguous() for key, tensor in repeat_targets.items()}
        kwargs.update(shared_arguments)

        # Apply function on the slice and restore the dimension for concatenation.
        for key, tensor in function(**kwargs).items():
            if key in shared_keys:
                continue

            if key not in output:
                output[key] = []

            output[key].append(tensor.unsqueeze(dim=dim))

    # Check whether the outputs are have the same size.
    assert all(len(t) == size for t in output.values())

    # Concatenate all outputs, and return.
    return {key: torch.cat(tensor, dim=dim).contiguous() for key, tensor in output.items()}


def shift_target(target: torch.Tensor, fill_value=PAD_ID) -> torch.Tensor:
    """
    Shift matrix to build generation targets.

    :param torch.Tensor target: Target tensor to build generation targets. Shape [B, T]
    :param fill_value: Value to be filled at the padded positions.
    :rtype: torch.Tensor
    :return: Tensor with shape [B, T], where (i, j)-entries are (i, j+1) entry of target tensor.
    """
    # Target does not require gradients.
    with torch.no_grad():
        pad_at_end = torch.full((target.shape[0], 1), fill_value=fill_value, dtype=target.dtype, device=target.device)
        return torch.cat([target[:, 1:], pad_at_end], dim=-1).contiguous()


def get_embedding_without_pad(embedding: Union[nn.Embedding, torch.Tensor],
                              tokens: torch.Tensor, ignore_index=PAD_ID) -> torch.Tensor:
    """
    Get embedding vectors of given token tensor with ignored indices are zero-filled.

    :param nn.Embedding embedding: An embedding instance
    :param torch.Tensor tokens: A Long Tensor to build embedding vectors.
    :param int ignore_index: Index to be ignored. `PAD_ID` by default.
    :rtype: torch.Tensor
    :return: Embedding vector of given token tensor.
    """
    # Clone tokens and fill masked values as zeros.
    tokens = tokens.clone()
    ignore_positions = (tokens == ignore_index)
    if ignore_positions.any():
        tokens.masked_fill_(ignore_positions, 0)

    # Apply embedding matrix
    if isinstance(embedding, nn.Embedding):
        embedding = embedding(tokens)
    else:
        embedding = F.embedding(tokens, embedding)

    # Set masked values as zero vector.
    if ignore_positions.any():
        embedding.masked_fill_(ignore_positions.unsqueeze(-1), 0.0)

    return embedding.contiguous()


__all__ = ['apply_module_dict', 'apply_across_dim', 'shift_target', 'get_embedding_without_pad']
