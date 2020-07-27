import torch
from torch import nn
from torch.nn.modules.transformer import TransformerDecoderLayer
from transformers.modeling_bert import gelu_new as gelu_bert

from page.config import ModelConfig
from page.const import *


class MultiheadAttentionWeights(nn.Module):
    """
    Class for computing multi-head attention weights (follows the paper, 'Attention is all you need')

    This class computes dot-product between query Q and key K, i.e.

    .. math::
        \\frac{Q^\\top K}{\\sqrt{D}}
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttentionWeights class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default.
        :keyword int num_heads: Number of attention heads (N). 12 by default.
        """
        super().__init__()
        self.config = config

        # Check whether D is divisible by H.
        assert self.hidden_dim % self.num_heads == 0, \
            "Hidden dimension %s is not divisible by the number of heads %s." % (self.hidden_dim, self.num_heads)

        # Linear transform for query Q
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Linear transform for key K
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Vector dimension D of input of a single attention head
        self.dim_head = self.hidden_dim // self.num_heads
        # Square root of vector dimension, i.e. \\sqrt{D}
        self.sqrt_dim = self.dim_head ** 0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, head_at_last: bool = True) -> torch.Tensor:
        """
        Compute multi-head attention weights

        :param torch.Tensor query:
            FloatTensor representing the query matrix Q with shape [B, S, H],
            where B = batch size, S = query sequence length, and H = vector dimension of hidden states.
        :param torch.Tensor key:
            FloatTensor representing the key matrix K with shape [B, T, H] or [1, T, H], where T = key sequence length
            By default, this is `None` (Use query matrix Q as a key matrix)
        :param torch.Tensor key_ignorance_mask:
            BoolTensor representing the mask for ignoring column vector in matrix K, with shape [B, T].
            If an element at (b, t) is `True,` then all return elements at B=b, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param torch.Tensor attention_mask:
            BoolTensor representing Attention mask for ignoring a key for each query item, with shape [S, T].
            If an element at (s, t) is `True,` then all return elements at S=s, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param bool head_at_last:
            Use `True` to make shape of return value be [B, S, T, N], where N = number of attention heads.
            If `False,` this method will return [B, N, S, T].
            By default, this is `True`
        :rtype: torch.FloatTensor
        :return: FloatTensor of Multi-head Attention weights
        """

        # If key is None, reuse query matrix Q.
        if key is None:
            key = query

        # Check size & type conditions
        assert query.shape[0] == key.shape[0] or key.shape[0] == 1 or query.shape[0] == 1
        assert key_ignorance_mask is None or (key.shape[:2] == key_ignorance_mask.shape and
                                              key_ignorance_mask.dtype == torch.bool)
        assert attention_mask is None or (query.shape[1] == attention_mask.shape[0] and
                                          key.shape[1] == attention_mask.shape[1] and
                                          attention_mask.dtype == torch.bool)

        # Store length information
        query_len = query.shape[1]
        key_len = key.shape[1]
        batch_size = max(key.shape[0], query.shape[0])

        # Project query & key with linear transformations
        query = self.linear_q(query)
        key = self.linear_k(key)

        # Scale query with sqrt(dim)
        query = query / self.sqrt_dim

        # If key / value has shape [1, T, H], expand it.
        if query.shape[0] == 1:
            query = query.expand(batch_size, -1, -1)
        if key.shape[0] == 1:
            key = key.expand(batch_size, -1, -1)

        # Transform query [B, S, N, H/N] -> [B, N, S, H/N] -> [BN, S, H/N].
        query = query.view(batch_size, query_len, self.num_heads, self.dim_head) \
            .transpose(1, 2).flatten(0, 1).contiguous()
        # Transform key [B, T, N, H/N] -> [B, N, H/N, T] -> [BN, H/T, T].
        key = key.view(batch_size, key_len, self.num_heads, self.dim_head) \
            .permute(0, 2, 3, 1).flatten(0, 1).contiguous()

        # Compute attention weights: [BN, S, T] -> [B, N, S, T]
        attention_weights = torch.bmm(query, key).view(batch_size, self.num_heads, query_len, key_len).contiguous()

        # Apply masks (IMPORTANT!!! This should be applied after GELU for output weights)
        if attention_mask is not None:
            # Recap: attention mask has shape [S, T], which can be broadcasted as [1, 1, S, T].
            attention_weights.masked_fill_(attention_mask, NEG_INF)

        if key_ignorance_mask is not None:
            # Recap: ignorance mask has shape [B, T] -> [B, 1, 1, T] and apply it.
            attention_weights.masked_fill_(key_ignorance_mask.unsqueeze(1).unsqueeze(1), NEG_INF)

        if head_at_last:
            # Output will be [B, N, S, T] -> [B, S, T, N]
            return attention_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attention_weights

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of hidden states (H)
        """
        return self.config.get('hidden_dim', 768)

    @property
    def num_heads(self) -> int:
        """
        :rtype: int
        :return: Number of attention heads (N)
        """
        return self.config.get('num_heads', 12)


class MultiheadAttention(nn.Module):
    """
    Class for computing multi-head attention (follows the paper, 'Attention is all you need')

    This class computes attention over K-V pairs with query Q, i.e.

    .. math::
        \\textrm{softmax}\\left(\\frac{Q^\\top K}{\\sqrt{D}}\\right) V
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttention class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default
        :keyword int num_heads: Number of attention heads (N). 12 by default
        :keyword float dropout_p: Probability of dropout. 0 by default
        """
        super().__init__()
        # Multi-head Attention Weight layer
        self.attn = MultiheadAttentionWeights(**config)
        # Dropout over attention weights (as in 'Attention is all you need')
        self.dropout_attn = nn.Dropout(self.dropout_p)
        # Linear transformations for value and output matrix.
        self.linear_v = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)
        self.linear_out = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, return_weights: bool = False, **kwargs):
        """
        Compute multi-head attention

        :param torch.Tensor query:
            FloatTensor representing the query matrix Q with shape [B, S, H],
            where B = batch size, S = query sequence length, and H = vector dimension of hidden states.
        :param torch.Tensor key_value:
            FloatTensor representing the key matrix K or value matrix V with shape [B, T, H] or [1, T, H],
            where T = key sequence length.
            By default, this is `None` (Use query matrix Q as a key matrix)
        :param torch.Tensor key_ignorance_mask:
            BoolTensor representing the mask for ignoring column vector in matrix K, with shape [B, T].
            If an element at (b, t) is `True,` then all return elements at B=b, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param torch.Tensor attention_mask:
            BoolTensor representing Attention mask for ignoring a key for each query item, with shape [S, T].
            If an element at (s, t) is `True,` then all return elements at S=s, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param bool return_weights:
            Use `True` to return attention weights. By default, this is `True.`
        :rtype: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
        :return:
            If head_at_last is True, return (Attention Output, Attention Weights).
            Otherwise, return only the Attention Output
            - Attention Output: Shape [B, S, H].
            - Attention Weights: Shape [B, S, T, N].
        """
        # If key_value is None, reuse query matrix Q.
        if key_value is None:
            key_value = query

        # Compute attention scores: [B, N, S, T].
        attn_weights = self.attn(query=query, key=key_value, key_ignorance_mask=key_ignorance_mask,
                                 attention_mask=attention_mask, head_at_last=False)

        # Retrive shape
        batch_size, _, query_len, key_len = attn_weights.shape

        # Compute Softmax values. Shape [B, N, S, T] -> [BN, S, T].
        # For numerical stability, replace NaN with -Inf. (NaN occurs when we should ignore all weights.)
        attn = attn_weights.softmax(dim=-1)
        attn = self.dropout_attn(attn)  # Dropout was applied after softmax in the original paper.
        attn = attn.masked_fill(torch.isnan(attn), 0.0).view(-1, query_len, key_len)

        # Pass linear and transpose value matrix: [1 or B, T, N, H/N] -> [1 or B, N, T, H/N].
        value_size = key_value.shape[0]
        value = self.linear_v(key_value) \
            .view(value_size, key_len, self.attn.num_heads, self.attn.dim_head).transpose(1, 2)

        # If value has shape [1, *], expand it.
        if value_size == 1:
            value = value.expand(batch_size, -1, -1, -1)

        # Flatten dim #0 and #1: [B, N, T, H/N] -> [BN, T, H/N].
        value = value.flatten(0, 1).contiguous()

        # Compute output of weighted sum: [BN, S, H/N] -> [B, N, S, H/N] -> [B, S, N, H/N] -> [B, S, H].
        output = torch.bmm(attn, value) \
            .view(batch_size, self.attn.num_heads, query_len, self.attn.dim_head) \
            .transpose(1, 2).flatten(2, 3).contiguous()

        # Map outputs and return. [B, S, H].
        output = self.linear_out(output)

        if return_weights:
            return output, attn_weights.permute(0, 2, 3, 1).contiguous()
        else:
            # Map outputs and return. [B, S, H].
            return output

    @property
    def dropout_p(self) -> float:
        """
        :rtype: float
        :return: Probability of dropout. 0 by default
        """
        return self.attn.config.get('dropout', 0.0)


class TransformerLayer(nn.Module):
    """
    Class for Transformer Encoder/Decoder layer (follows the paper, 'Attention is all you need')
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize TransformerLayer class

        :param ModelConfig config: Configuration of this Encoder/Decoder layer
        """
        super().__init__()

        # Self-attention layer
        self.attn = MultiheadAttention(hidden_dim=config.hidden_dim, num_heads=config.num_decoder_heads,
                                       layernorm_eps=config.layernorm_eps, dropout=config.dropout_attn)
        # Source-Target attention layer
        self.mem = MultiheadAttention(hidden_dim=config.hidden_dim, num_heads=config.num_decoder_heads,
                                      layernorm_eps=config.layernorm_eps, dropout=config.dropout_attn)

        # Dropout for self-attention
        self.dropout_attn = nn.Dropout(config.dropout_layer)
        # Dropout for source-target attention
        self.dropout_mem = nn.Dropout(config.dropout_layer)
        # Dropout for expansion before outputting
        self.dropout_expand = nn.Dropout(config.dropout_layer)
        # Dropout for outputting
        self.dropout_out = nn.Dropout(config.dropout_layer)

        # Linear transformation layer for expansion (H -> I) where I = vector dimension of intermediate state
        self.lin_expand = nn.Linear(config.hidden_dim, config.intermediate_dim)
        # Linear transformation layer for output (I -> H)
        self.lin_collapse = nn.Linear(config.intermediate_dim, config.hidden_dim)

        # Post Layer Normalization for self-attention
        self.norm_attn = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        # Post Layer Normalization for source-target attention
        self.norm_mem = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        # Post Layer Normalization for outputting
        self.norm_out = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)

    def forward(self, target, target_ignorance_mask=None, target_attention_mask=None,
                memory=None, memory_ignorance_mask=None):
        """
        Forward-computation of Transformer Encoder/Decoder layers

        :param torch.Tensor target:
            FloatTensor indicating Sequence of target vectors. Shape [B, T, H]
            where B = batch size, T = length of target sequence, H = vector dimension of hidden state
        :param torch.Tensor target_ignorance_mask:
            BoolTensor indicating Mask for target tokens that should be ignored. Shape [B, T].
        :param torch.Tensor target_attention_mask:
            BoolTensor indicating Target-to-target Attention mask for target tokens. Shape [T, T].
        :param torch.Tensor memory:
            FloatTensor indicating Sequence of source vectors. Shape [B, S, H]
            where S = length of source sequence
            This can be None when you want to use this layer as an encoder layer.
        :param torch.Tensor memory_ignorance_mask:
            BoolTensor indicating Mask for source tokens that should be ignored. Shape [B, S].
        :rtype: torch.FloatTensor
        :return: Decoder hidden states per each target token, shape [B, S, H].
        """
        # Compute self-attention
        attented = self.attn(query=target, attention_mask=target_attention_mask,
                             key_ignorance_mask=target_ignorance_mask)
        target = target + self.dropout_attn(attented)
        target = self.norm_attn(target)

        # Compute attention over targets with source as queries.
        if memory is not None:
            attented = self.mem(query=target, key_value=memory, key_ignorance_mask=memory_ignorance_mask)
            target = target + self.dropout_mem(attented)
            target = self.norm_mem(target)

        # Pass linear transformations
        output = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(target))))
        target = target + self.dropout_out(output)
        target = self.norm_out(target)

        return target


class WrappedMultiheadAttention(nn.MultiheadAttention):
    """
    Class for computing multi-head attention (using PyTorch implementation)
    This class inherits torch.nn.MultiheadAttention class.
    """

    def __init__(self, batch_first=True, **config):
        """
        Initialize WrappedMultiheadAttention class

        :param bool batch_first: True if you want to make this return batch-first Tensors. Default is True.
        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default
        :keyword int num_heads: Number of attention heads (N). 12 by default
        :keyword float dropout_p: Probability of dropout. 0 by default
        """
        super().__init__(embed_dim=config.get('hidden_dim', 768),
                         num_heads=config.get('num_heads', 12),
                         dropout=config.get('dropout', 0),
                         bias=True, add_bias_kv=False)

        self.config = config
        self.batch_first = batch_first

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, return_weights: bool = False, **kwargs):
        """
        Compute multi-head attention

        :param torch.Tensor query:
            FloatTensor representing the query matrix Q.
            Shape [B, S, H] if self.batch_first is True, otherwise [S, B, H],
            where B = batch size, S = query sequence length, and H = vector dimension of hidden states.
        :param torch.Tensor key_value:
            FloatTensor representing the key matrix K or value matrix V.
            Shape [B, T, H] if self.batch_first is True, otherwise [S, T, H],
            where T = key sequence length.
            By default, this is `None` (Use query matrix Q as a key matrix)
        :param torch.Tensor key_ignorance_mask:
            BoolTensor representing the mask for ignoring column vector in matrix K, with shape [B, T].
            If an element at (b, t) is `True,` then all return elements at B=b, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param torch.Tensor attention_mask:
            BoolTensor representing Attention mask for ignoring a key for each query item, with shape [S, T].
            If an element at (s, t) is `True,` then all return elements at S=s, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param bool return_weights:
            Use `True` to return attention weights. By default, this is `True.`
        :rtype: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
        :return:
            If head_at_last is True, return (Attention Output, Attention Weights).
            Otherwise, return only the Attention Output
            - Attention Output: Shape [B, S, H] if batch_first is True, otherwise [S, B, H]
            - Attention Weights: Shape [B, S, T, N].
        """
        # Copy query if key_value is not provided.
        key = key_value if key_value is not None else query

        if attention_mask is not None:
            # Target attention mask is a bool tensor, but Pytorch implementation requires a float tensor.
            attention_mask = torch.zeros_like(attention_mask, dtype=torch.float) \
                .masked_fill_(attention_mask, NEG_INF)

        if self.batch_first:
            # Transpose [B, T, H] and [B, S, H] to [T, B, H] and [S, B, H].
            query_batch_second = query.transpose(0, 1)
            key_batch_second = key.transpose(0, 1)
            result = super().forward(query=query_batch_second, key=key_batch_second, value=key_batch_second,
                                     key_padding_mask=key_ignorance_mask,
                                     attn_mask=attention_mask, need_weights=return_weights)

            # Transpose the result [S, B, H] to [B, S, H]
            result = result[0].transpose(0, 1), result[1]
        else:
            result = super().forward(query=query, key=key, value=key, key_padding_mask=key_ignorance_mask,
                                     attn_mask=attention_mask, need_weights=return_weights)

        if return_weights:
            return result
        else:
            return result[0]


class WrappedTransformerLayer(TransformerDecoderLayer):
    """
    Class for computing Transformer Decoder layer (using Pytorch implementation)
    This class inherits torch.nn.modules.transformer.TransformerDecoderLayer class.
    """

    def __init__(self, config: ModelConfig, batch_first=True):
        """
        Initialize WrappedTransformerLayer class

        :param ModelConfig config: Configuration of entire model
        :param bool batch_first: True if you want to make this return batch-first Tensors. Default is True.
        """
        super().__init__(d_model=config.hidden_dim,
                         nhead=config.num_decoder_heads,
                         dim_feedforward=config.intermediate_dim,
                         dropout=config.dropout_layer,
                         activation='relu')

        self.batch_first = batch_first

        # Replace activation to GeLU
        self.activation = gelu_bert

        # Copy LayerNorm's epsilon value
        eps = config.layernorm_eps
        setattr(self.norm1, 'eps', eps)
        setattr(self.norm2, 'eps', eps)
        setattr(self.norm3, 'eps', eps)

    def forward(self, target: torch.Tensor, memory: torch.Tensor, target_attention_mask: torch.Tensor = None,
                target_ignorance_mask: torch.Tensor = None, memory_ignorance_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        Forward-computation of Transformer Encoder/Decoder layers

        :param torch.Tensor target:
            FloatTensor indicating Sequence of target vectors. Shape [B, T, H]
            where B = batch size, T = length of target sequence, H = vector dimension of hidden state
        :param torch.Tensor target_ignorance_mask:
            BoolTensor indicating Mask for target tokens that should be ignored. Shape [B, T].
        :param torch.Tensor target_attention_mask:
            BoolTensor indicating Target-to-target Attention mask for target tokens. Shape [T, T].
        :param torch.Tensor memory:
            FloatTensor indicating Sequence of source vectors. Shape [B, S, H]
            where S = length of source sequence
            This can be None when you want to use this layer as an encoder layer.
        :param torch.Tensor memory_ignorance_mask:
            BoolTensor indicating Mask for source tokens that should be ignored. Shape [B, S].
        :rtype: torch.FloatTensor
        :return: Decoder hidden states per each target token, shape [B, S, H].
        """
        if target_attention_mask is not None:
            # Target attention mask is a bool tensor, but Pytorch implementation requires a float tensor.
            target_attention_mask = torch.zeros_like(target_attention_mask, dtype=torch.float) \
                .masked_fill_(target_attention_mask, NEG_INF)

        if self.batch_first:
            # Transpose [B, T, H] and [B, S, H] to [T, B, H] and [S, B, H].
            target_batch_second = target.transpose(0, 1)
            memory_batch_second = memory.transpose(0, 1)

            result_batch_second = super().forward(tgt=target_batch_second, memory=memory_batch_second,
                                                  tgt_mask=target_attention_mask,
                                                  tgt_key_padding_mask=target_ignorance_mask,
                                                  memory_key_padding_mask=memory_ignorance_mask)

            # Transpose the result [S, B, H] to [B, S, H]
            return result_batch_second.transpose(0, 1)
        else:
            return super().forward(tgt=target, memory=memory, tgt_mask=target_attention_mask,
                                   tgt_key_padding_mask=target_ignorance_mask,
                                   memory_key_padding_mask=memory_ignorance_mask)


__all__ = ['MultiheadAttentionWeights', 'MultiheadAttention', 'TransformerLayer',
           'WrappedMultiheadAttention', 'WrappedTransformerLayer']
