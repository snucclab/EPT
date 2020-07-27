from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from page.config import ModelConfig
from page.const import *
from .attention import *
from .embed import PositionalEncoding
from .layer import *
from .loss import loss_and_accuracy
from .util import *


def mask_forward(sz: int, diagonal: int = 1) -> torch.Tensor:
    """
    Generate a mask that ignores future words. Each (i, j)-entry will be True if j >= i + diagonal

    :param int sz: Length of the sequence.
    :param int diagonal: Amount of shift for diagonal entries.
    :rtype: torch.Tensor
    :return: Mask tensor with shape [sz, sz].
    """
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal=diagonal).contiguous()


class DecoderModel(nn.Module):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, config: ModelConfig):
        """
        Initiate Equation Builder instance

        :param ModelConfig config: Configuration of this model
        """
        super().__init__()
        # Save configuration.
        self.config = config

    @property
    def embedding_dim(self) -> int:
        """
        :rtype: int
        :return: Dimension of embedding vector
        """
        return self.config.embedding_dim

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Dimension of hidden vector.
        """
        return self.config.hidden_dim

    @property
    def num_hidden_layers(self) -> int:
        """
        :rtype: int
        :return: Number of repetition for applying the same transformer layer
        """
        return self.config.num_decoder_layers

    @property
    def init_factor(self) -> float:
        """
        :rtype: float
        :return: Standard deviation of normal distribution that will be used for initializing weights.
        """
        return self.config.init_factor

    @property
    def layernorm_eps(self) -> float:
        """
        :rtype: float
        :return: Epsilon to avoid zero-division in LayerNorm.
        """
        return self.config.layernorm_eps

    @property
    def num_heads(self) -> int:
        """
        :rtype: int
        :return: Number of heads in a transformer layer.
        """
        return self.config.num_decoder_heads

    @property
    def num_pointer_heads(self) -> int:
        """
        :rtype: int
        :return: Number of heads in the last pointer layer.
        """
        return self.config.num_pointer_heads

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        raise NotImplementedError()

    @property
    def is_expression_type(self) -> bool:
        """
        :rtype: bool
        :return: True if this model requires Expression type sequence
        """
        return self.required_field in [FIELD_EXPR_PTR, FIELD_EXPR_GEN]

    def save_pretrained(self, save_directory: str):
        """
        Save current state of Equation Builder.

        :param str save_directory: String that represents path to the directory where this will be saved.
        """
        # Write state dictionary
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), Path(save_directory, '%s.pt' % self.__class__.__name__))

    @classmethod
    def from_pretrained(cls, config: ModelConfig):
        """
        Load pre-trained model

        :param ModelConfig config: Configuration of a model that will be loaded
        :rtype: DecoderModel
        :return: A DecoderModel instance
        """
        # Import the model if available, otherwise create it using keyword argument
        model = cls(config)
        if config.chkpt_path is not None:
            model.load_state_dict(torch.load(Path(config.chkpt_path, '%s.pt' % cls.__name__)))

        return model

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights

        :param nn.Module module: Module to be initialized.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            # nn.Linear has 'weight' and 'bias', nn.Embedding has 'weight',
            # and nn.MultiheadAttention has *_weight and *_bias
            for name, param in module.named_parameters():
                if param is None:
                    continue

                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=self.init_factor)
                elif 'bias' in name:
                    param.data.zero_()
                else:
                    raise NotImplementedError("This case is not considered!")
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization as an identity funciton.
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_single(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of computed values
        """
        raise NotImplementedError()

    def _build_target_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
        """
        raise NotImplementedError()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of decoder model

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of tensors
            - If this model is currently on training phase, values will be accuracy or loss tensors
            - Otherwise, values will be tensors representing predicted distribution of output
        """
        result = {}
        if self.training:
            # Forward single beam input.
            output = self._forward_single(**kwargs)
            # Build targets
            with torch.no_grad():
                targets = self._build_target_dict(**kwargs)

            for key in targets:
                result.update(loss_and_accuracy(output[key], shift_target(targets[key]), prefix='Train_%s' % key))
        else:
            with torch.no_grad():
                result.update(apply_across_dim(self._forward_single, dim=1,
                                               shared_keys={IN_TXT, IN_TPAD, IN_TNUM, IN_TNPAD}, **kwargs))

        return result


class ExpressionDecoderModel(DecoderModel):
    """
    Decoding model that generates expression sequences (Abstract class)
    """

    def __init__(self, config):
        super().__init__(config)

        """ Embedding layers """
        # Look-up table E_f(.) for operator embedding vectors (in Equation 2)
        self.operator_word_embedding = nn.Embedding(self.operator_vocab_size, self.hidden_dim)
        # Positional encoding PE(.) (in Equation 2, 5)
        self.operator_pos_embedding = PositionalEncoding(self.hidden_dim)
        # Vectors representing source: u_num, u_const, u_expr in Equation 3, 4, 5
        self.operand_source_embedding = nn.Embedding(3, self.hidden_dim)

        """ Scalar parameters """
        # Initial degrading factor value for c_f and c_a.
        degrade_factor = self.embedding_dim ** 0.5
        # c_f in Equation 2
        self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        # c_a in Equation 3, 4, 5
        self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Layer Normalizations """
        # LN_f in Equation 2
        self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        # LN_a in Equation 3, 4, 5
        self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        """ Linear Transformation """
        # Linear transformation from embedding space to hidden space: FF_in in Equation 1.
        self.embed_to_hidden = nn.Linear(self.hidden_dim * (self.max_arity + 1), self.hidden_dim)

        """ Transformer layer """
        # Shared transformer layer for decoding (TransformerDecoder in Figure 2)
        self.shared_decoder_layer = TransformerLayer(config)

        """ Output layer """
        # Linear transformation from hidden space to pseudo-probability space: FF_out in Equation 6
        self.operator_out = nn.Linear(self.hidden_dim, self.operator_vocab_size)
        # Softmax layers, which can handle infinity values properly (used in Equation 6, 10)
        self.softmax = LogSoftmax(dim=-1)

        # Argument output will be defined in sub-classes
        # Initialize will be done in sub-classes

    @property
    def operator_vocab_size(self) -> int:
        """
        :rtype: int
        :return: Size of operator vocabulary
        """
        return self.config.operator_vocab_size

    @property
    def operand_vocab_size(self) -> int:
        """
        :rtype: int
        :return: Size of operand vocabulary including variables (used in VanillaTransformer + Expression Model)
        """
        return self.config.operand_vocab_size

    @property
    def constant_vocab_size(self) -> int:
        """
        :rtype: int
        :return: Size of constant vocabulary (used in EPT)
        """
        return self.config.constant_vocab_size

    @property
    def max_arity(self) -> int:
        """
        :rtype: int
        :return: Maximum possible arity of an operator
        """
        # This could be static. We defined it as an property for the future.
        return max([op['arity'] for op in OPERATORS.values()], default=2)

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor ids:
            LongTensor containing index-type information of operands. (This corresponds to a_ij in the paper)
        :param torch.Tensor mem_pos:
            FloatTensor containing positional encoding used so far. (i.e. PE(.) in the paper)
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
        :rtype: torch.Tensor
        :return: A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Compute input of the decoder, i.e. Equation 1 in the paper.

        :param torch.Tensor ids:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H],
            where N = maximum number of written numbers in the batch, and H = dimension of hidden state.
        :rtype: torch.Tensor
        :return: A FloatTensor representing input vector v_i in Equation 1. Shape [B, T, H].
        """
        # Operator embedding: [B, T, H] (Equation 2)
        # - compute E_f first
        operator = get_embedding_without_pad(self.operator_word_embedding, ids.select(dim=-1, index=0))
        # - compute PE(.): [T, H]
        operator_pos = self.operator_pos_embedding(ids.shape[1])
        # - apply c_f and layer norm, and reshape it as [B, T, 1, H]
        operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0)).unsqueeze(2)

        # Operand embedding [B, T, A, H] (Equation 3, 4, 5)
        # - compute c_a u_* first.
        operand = get_embedding_without_pad(self.operand_source_embedding, ids[:, :, 1::2]) * self.operand_source_factor
        # - add operand embedding
        operand += self._build_operand_embed(ids, operator_pos, nums)
        # - apply layer norm
        operand = self.operand_norm(operand)

        # Concatenate embedding: [B, T, 1+A, H] -> [B, T, (1+A)H]
        operator_operands = torch.cat([operator, operand], dim=2).contiguous().flatten(start_dim=2)
        # Do linear transformation (Equation 1)
        return self.embed_to_hidden(operator_operands)

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor = None,
                               text: torch.Tensor = None, text_pad: torch.Tensor = None) -> torch.Tensor:
        """
        Compute decoder's hidden state vectors, i.e. d_i in the paper

        :param torch.Tensor embedding:
            FloatTensor containing input vectors v_i. Shape [B, T, H],
            where B = batch size, T = length of decoding sequence, and H = dimension of input embedding
        :param torch.Tensor embedding_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence
            Shape [B, T]
        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where S = length of input sequence.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :rtype: torch.Tensor
        :return: A FloatTensor of shape [B, T, H], which contains decoder's hidden states.
        """
        # Build forward mask
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        # Repeatedly pass TransformerDecoder layer
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_decoder_layer(target=output, memory=text, target_attention_mask=mask,
                                               target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)

        return output

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        equation: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where B = batch size, T = length of input sequence, and H = dimension of input embedding.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :param torch.Tensor text_num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H], where N = maximum number of written numbers in the batch.
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - 'operator': Log probability of next operators (i.e. Equation 6 without argmax).
                FloatTensor with shape [B, T, F], where F = size of operator vocabulary.
            - '_out': Decoder's hidden states. FloatTensor with shape [B, T, H]
            - '_not_usable': Indicating positions that corresponding output values are not usable in the operands.
                BoolTensor with Shape [B, T].
        """
        # Embedding: [B, T, H]
        operator_ids = equation.select(dim=2, index=0)
        output = self._build_decoder_input(ids=equation, nums=text_num)
        output_pad = operator_ids == PAD_ID

        # Ignore the result of equality at the function output
        output_not_usable = output_pad.clone()
        output_not_usable[:, :-1].masked_fill_(operator_ids[:, 1:] == FUN_EQ_SGN_ID, True)
        # We need offset '1' because 'function_word' is input and output_not_usable is 1-step shifted output.

        # Decoder output: [B, T, H]
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)

        # Compute function output (with 'NEW_EQN' masked)
        operator_out = self.operator_out(output)

        if not self.training:
            operator_out[:, :, FUN_NEW_EQN_ID] = NEG_INF
            # Can end after equation formed, i.e. END_EQN is available when the input is EQ_SGN.
            operator_out[:, :, FUN_END_EQN_ID].masked_fill_(operator_ids != FUN_EQ_SGN_ID, NEG_INF)

        # Predict function output.
        result = {'operator': self.softmax(operator_out),
                  '_out': output, '_not_usable': output_not_usable}

        # Remaining work will be done by subclasses
        return result


class ExpressionTransformer(ExpressionDecoderModel):
    """
    Vanilla Transformer + Expression (The second ablated model)
    """

    def __init__(self, config):
        super().__init__(config)

        """ Operand embedding """
        # Look-up table for embedding vectors: E_c used in Vanilla Transformer + Expression (Appendix)
        self.operand_word_embedding = nn.Embedding(self.operand_vocab_size, self.hidden_dim)

        """ Output layer """
        # Linear transformation from hidden space to operand output space:
        # FF_j used in Vanilla Transformer + Expression (Appendix)
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_out': nn.Linear(self.hidden_dim, self.operand_vocab_size)
            }) for _ in range(self.max_arity)
        ])

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return FIELD_EXPR_GEN

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor ids:
            LongTensor containing source-content information of operands. (This corresponds to a_ij in the paper)
            Shape [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor mem_pos:
            FloatTensor containing positional encoding used so far. (i.e. PE(.) in the paper)
            Shape [B, T, H], where H = dimension of hidden state
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H], where N = Maximum number of written numbers in the batch.
        :rtype: torch.Tensor
        :return:
            A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
            Shape [B, T, A, H]
        """
        # Compute operand embedding (Equation 4 in the paper and 3-rd and 4-th Equation in the appendix)
        # Adding u vectors will be done in _build_decoder_input.
        # We will ignore information about the source (slice 1::2) and operator (index 0).
        return get_embedding_without_pad(self.operand_word_embedding, ids[:, :, 2::2])

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None,
                        text_num: torch.Tensor = None, text_numpad: torch.Tensor = None,
                        equation: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where B = batch size, T = length of input sequence, and H = dimension of input embedding.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :param torch.Tensor text_num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H], where N = maximum number of written numbers in the batch.
        :param torch.Tensor text_numpad:
            BoolTensor, whose values are True if corresponding position is PAD in the number sequence
            Shape [B, N]
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - 'operator': Log probability of next operators (i.e. Equation 6 without argmax).
                FloatTensor with shape [B, T, F], where F = size of operator vocabulary.
            - 'operand_J': Log probability of next J-th operands (i.e. Equation 10 without argmax).
                FloatTensor with shape [B, T, V], where V = size of operand vocabulary.
        """
        # Retrieve decoder's hidden states
        # Dictionary will have 'func', '_out', and '_not_usable'
        result = super()._forward_single(text, text_pad, text_num, equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden state: [B, T, H]
        output = result.pop('_out')
        # Mask indicating whether expression can be used as an operand: [B, T] -> [B, 1, T]
        output_not_usable = result.pop('_not_usable').unsqueeze(1)
        # Forward mask: [T, T] -> [1, T, T]
        forward_mask = mask_forward(output.shape[1], diagonal=0).unsqueeze(0).to(output.device)

        # Number tokens are placed on 1:1+NUM_MAX
        num_begin = 1
        num_used = num_begin + min(text_num.shape[1], NUM_MAX)
        num_end = num_begin + NUM_MAX
        # Memory tokens are placed on 1+NUM_MAX:1+NUM_MAX+MEM_MAX
        mem_used = num_end + min(output.shape[1], MEM_MAX)
        mem_end = num_end + MEM_MAX

        # Predict arguments
        for j, layer in enumerate(self.operand_out):
            word_output = apply_module_dict(layer, encoded=output)

            # Mask probabilities when evaluating.
            if not self.training:
                # Ignore probabilities on not-appeared number tokens
                word_output[:, :, num_begin:num_used].masked_fill_(text_numpad.unsqueeze(1), NEG_INF)
                word_output[:, :, num_used:num_end] = NEG_INF

                # Ignore probabilities on non-appeared memory tokens
                word_output[:, :, num_end:mem_used].masked_fill_(output_not_usable, NEG_INF)
                word_output[:, :, num_end:mem_used].masked_fill_(forward_mask, NEG_INF)
                word_output[:, :, mem_used:mem_end] = NEG_INF

            # Apply softmax after masking (compute 'operand_J')
            result['operand_%s' % j] = self.softmax(word_output)

        return result

    def _build_target_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
            - 'operator': Index of next operators (i.e. Equation 6).
                LongTensor with shape [B, T].
            - 'operand_J': Index of next J-th operands (i.e. Equation 10).
                LongTensor with shape [B, T].
        """
        # Build targets
        equation = kwargs[IN_EQN]
        targets = {'operator': equation.select(dim=-1, index=0)}
        for j in range(self.max_arity):
            targets['operand_%s' % j] = equation[:, :, (j * 2 + 2)]

        return targets


class ExpressionPointerTransformer(ExpressionDecoderModel):
    """
    The EPT model
    """

    def __init__(self, config):
        super().__init__(config)

        """ Operand embedding """
        # Look-up table for constants: E_c used in Equation 4
        self.constant_word_embedding = nn.Embedding(self.constant_vocab_size, self.hidden_dim)

        """ Output layer """
        # Group of layers to compute Equation 8, 9, and 10
        self.operand_out = nn.ModuleList([
            nn.ModuleDict({
                '0_attn': MultiheadAttentionWeights(hidden_dim=self.hidden_dim, num_heads=self.num_pointer_heads),
                '1_mean': Squeeze(dim=-1) if self.num_pointer_heads == 1 else AveragePooling(dim=-1)
            }) for _ in range(self.max_arity)
        ])

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return FIELD_EXPR_PTR

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor ids:
            LongTensor containing source-content information of operands. (This corresponds to a_ij in the paper)
            Shape [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor mem_pos:
            FloatTensor containing positional encoding used so far. (i.e. PE(.) in the paper)
            Shape [B, T, H], where H = dimension of hidden state
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H], where N = Maximum number of written numbers in the batch.
        :rtype: torch.Tensor
        :return:
            A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
            Shape [B, T, A, H]
        """
        # Tensor ids has 1 vocabulary index of operator and A pair of (source of operand, vocabulary index of operand)
        # Source of operand (slice 1::2), shape [B, T, A]
        operand_source = ids[:, :, 1::2]
        # Index of operand (slice 2::2), shape [B, T, A]
        operand_value = ids[:, :, 2::2]

        # Compute for number operands: [B, T, A, E] (Equation 3)
        number_operand = operand_value.masked_fill(operand_source != ARG_NUM_ID, PAD_ID)
        operand = torch.stack([get_embedding_without_pad(nums[b], number_operand[b])
                               for b in range(ids.shape[0])], dim=0).contiguous()

        # Compute for constant operands: [B, T, A, E] (Equation 4)
        operand += get_embedding_without_pad(self.constant_word_embedding,
                                             operand_value.masked_fill(operand_source != ARG_CON_ID, PAD_ID))

        # Compute for prior-result operands: [B, T, A, E] (Equation 5)
        prior_result_operand = operand_value.masked_fill(operand_source != ARG_MEM_ID, PAD_ID)
        operand += get_embedding_without_pad(mem_pos, prior_result_operand)
        return operand

    def _build_attention_keys(self, num: torch.Tensor, mem: torch.Tensor, num_pad: torch.Tensor = None,
                              mem_pad: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Attention Keys by concatenating all items in Equation 7.

        :param torch.Tensor num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H],
            where B = Batch size, N = Maximum number of written numbers in the batch, and H = dimension of hidden states
        :param torch.Tensor mem:
            FloatTensor containing decoder's hidden states corresponding to prior expression outputs.
            (i.e. d_i in the paper)
            Shape [B, T, H], where T = length of prior expression outputs
        :param torch.Tensor num_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the number sequence
            Shape [B, N]
        :param torch.Tensor mem_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the target expression sequence
            Shape [B, T]
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :return: Triple of Tensors
            - [0] Keys (A_ij in the paper). Shape [B, C+N+T, H], where C = size of constant vocabulary.
            - [1] Mask for positions that should be ignored in keys. Shape [B, C+N+T]
            - [2] Forward Attention Mask to ignore future tokens in the expression sequence. Shape [T, C+N+T]
        """
        # Retrieve size information
        batch_sz = num.shape[0]
        const_sz = self.constant_vocab_size
        const_num_sz = const_sz + num.shape[1]

        # Order: Const, Number, Memory
        # Constant keys: [C, E] -> [1, C, H] -> [B, C, H]
        const_key = self.constant_word_embedding.weight.unsqueeze(0).expand(batch_sz, const_sz, self.hidden_dim)

        # Key: [B, C+N+T, H]
        key = torch.cat([const_key, num, mem], dim=1).contiguous()
        # Key ignorance mask: [B, C+N+T]
        key_ignorance_mask = torch.zeros(key.shape[:2], dtype=torch.bool, device=key.device)
        if num_pad is not None:
            key_ignorance_mask[:, const_sz:const_num_sz] = num_pad
        if mem_pad is not None:
            key_ignorance_mask[:, const_num_sz:] = mem_pad

        # Attention mask: [T, C+N+T], exclude self.
        attention_mask = torch.zeros(mem.shape[1], key.shape[1], dtype=torch.bool, device=key.device)
        attention_mask[:, const_num_sz:] = mask_forward(mem.shape[1], diagonal=0).to(key_ignorance_mask.device)

        return key, key_ignorance_mask, attention_mask

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None,
                        text_num: torch.Tensor = None, text_numpad: torch.Tensor = None,
                        equation: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where B = batch size, T = length of input sequence, and H = dimension of input embedding.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :param torch.Tensor text_num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H], where N = maximum number of written numbers in the batch.
        :param torch.Tensor text_numpad:
            BoolTensor, whose values are True if corresponding position is PAD in the number sequence
            Shape [B, N]
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - 'operator': Log probability of next operators (i.e. Equation 6 without argmax).
                FloatTensor with shape [B, T, F], where F = size of operator vocabulary.
            - 'operand_J': Log probability of next J-th operands (i.e. Equation 10 without argmax).
                FloatTensor with shape [B, T, V], where V = size of operand vocabulary.
        """
        # Retrieve decoder's hidden states
        # Dictionary will have 'func', '_out', and '_not_usable'
        result = super()._forward_single(text, text_pad, text_num, equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden states: [B, T, H]
        output = result.pop('_out')
        # Mask indicating whether expression can be used as an operand: [B, T] -> [B, 1, T]
        output_not_usable = result.pop('_not_usable')

        # Build attention keys by concatenating constants, written numbers and prior outputs (Equation 7)
        key, key_ign_msk, attn_msk = self._build_attention_keys(num=text_num, mem=output,
                                                                num_pad=text_numpad, mem_pad=output_not_usable)

        # Predict arguments (Equation 8, 9, 10)
        for j, layer in enumerate(self.operand_out):
            score = apply_module_dict(layer, encoded=output, key=key, key_ignorance_mask=key_ign_msk,
                                      attention_mask=attn_msk)
            result['operand_%s' % j] = self.softmax(score)

        return result

    def _build_target_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
            - 'operator': Index of next operators (i.e. Equation 6).
                LongTensor with shape [B, T].
            - 'operand_J': Index of next J-th operands (i.e. Equation 10).
                LongTensor with shape [B, T].
        """
        # Build targets
        equation = kwargs[IN_EQN]

        # Offset of written numbers
        num_offset = self.constant_vocab_size
        # Offset of prior expressions
        mem_offset = num_offset + kwargs[IN_TNUM].shape[1]

        # Build dictionary for targets
        targets = {'operator': equation.select(dim=-1, index=0)}
        for i in range(self.max_arity):
            # Source of the operand
            operand_source = equation[:, :, (i * 2 + 1)]
            # Value of the operand
            operand_value = equation[:, :, (i * 2 + 2)].clamp_min(0)

            # Add index offsets.
            # - PAD_ID will be PAD_ID (-1),
            # - constants will use range from 0 to C (number of constants; exclusive)
            # - numbers will use range from C to C + N (N = max_num)
            # - prior expressions will use range C + N to C + N + T
            operand_value += operand_source.masked_fill(operand_source == ARG_NUM_ID, num_offset) \
                .masked_fill_(operand_source == ARG_MEM_ID, mem_offset)

            # Assign target value of J-th operand.
            targets['operand_%s' % i] = operand_value

        return targets


class OpDecoderModel(DecoderModel):
    """
    Decoding model that generates Op(Operator/Operand) sequences (Abstract class)
    """

    def __init__(self, config):
        super().__init__(config)

        """ Embedding look-up tables """
        # Token-level embedding
        self.word_embedding = nn.Embedding(self.op_vocab_size, self.hidden_dim)
        # Positional encoding
        self.pos_embedding = PositionalEncoding(self.hidden_dim)
        # LayerNorm for normalizing word embedding vector.
        self.word_hidden_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        # Factor that upweights word embedding vector. (c_in in the Appendix)
        degrade_factor = self.hidden_dim ** 0.5
        self.pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Decoding layer """
        # Shared transformer layer for decoding
        self.shared_layer = TransformerLayer(config)

        # Output layer will be defined in sub-classes
        # Weight will be initialized by sub-classes

    @property
    def op_vocab_size(self) -> int:
        """
        :rtype: int
        :return: Size of Op vocabulary
        """
        return self.config.op_vocab_size

    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build Op embedding

        :param torch.Tensor ids:
            LongTensor containing source-content information of operands. (This corresponds to t_i in the Appendix)
            Shape [B, T], where B = batch size, and T = length of expression sequence.
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H], where N = Maximum number of written numbers in the batch.
        :rtype: torch.Tensor
        :return:
            A FloatTensor representing op embedding vector v_i in the 1st equation of the Appendix (before applying LN)
            Shape [B, T, H]
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Compute input of the decoder, i.e. the 1st equation in the Appendix.

        :param torch.Tensor ids:
            LongTensor containing op tokens (This corresponds to t_i in the appendix)
            Shape: [B, T], where B = batch size and T = length of expression sequence.
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H],
            where N = maximum number of written numbers in the batch, and H = dimension of hidden state.
        :rtype: torch.Tensor
        :return: A FloatTensor representing input vector v_i in the 1st equation in the Appendix. Shape [B, T, H].
        """
        # Positions: [T, E]
        pos = self.pos_embedding(ids.shape[1])
        # Word embeddings: [B, T, E]
        word = self._build_word_embed(ids, nums)
        # Return [B, T, E]
        return self.word_hidden_norm(word * self.pos_factor + pos.unsqueeze(0))

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor = None,
                               text: torch.Tensor = None, text_pad: torch.Tensor = None) -> torch.Tensor:
        """
        Compute decoder's hidden state vectors, i.e. d_i in the paper

        :param torch.Tensor embedding:
            FloatTensor containing input vectors v_i. Shape [B, T, H],
            where B = batch size, T = length of decoding sequence, and H = dimension of input embedding
        :param torch.Tensor embedding_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence
            Shape [B, T]
        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where S = length of input sequence.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :rtype: torch.Tensor
        :return: A FloatTensor of shape [B, T, H], which contains decoder's hidden states.
        """
        # Build forward mask
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        # Repeatedly pass TransformerDecoder layer
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_layer(target=output, memory=text, target_attention_mask=mask,
                                       target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)

        return output

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        equation: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where B = batch size, T = length of input sequence, and H = dimension of input embedding.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :param torch.Tensor text_num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H], where N = maximum number of written numbers in the batch.
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+2A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - '_out': Decoder's hidden states. FloatTensor with shape [B, T, H]
        """
        # Embedding: [B, T, H]
        output = self._build_decoder_input(ids=equation, nums=text_num.relu())
        output_pad = equation == PAD_ID

        # Decoder's hidden states: [B, T, H]
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        result = {'_out': output}

        # Remaining work will be done by subclasses
        return result


class VanillaOpTransformer(OpDecoderModel):
    """
    The vanilla Transformer model
    """

    def __init__(self, config):
        super().__init__(config)

        """ Op token Generator """
        self.op_out = nn.Linear(self.hidden_dim, self.op_vocab_size)
        self.softmax = LogSoftmax(dim=-1)

        """ Initialize weights """
        with torch.no_grad():
            # Initialize Linear, LayerNorm, Embedding
            self.apply(self._init_weights)

    @property
    def required_field(self) -> str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return FIELD_OP_GEN

    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        """
        Build Op embedding

        :param torch.Tensor ids:
            LongTensor containing source-content information of operands. (This corresponds to t_i in the Appendix)
            Shape [B, T], where B = batch size, and T = length of expression sequence.
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H], where N = Maximum number of written numbers in the batch.
        :rtype: torch.Tensor
        :return:
            A FloatTensor representing op embedding vector v_i in the 1st equation of the Appendix (before applying LN)
            Shape [B, T, H]
        """
        return get_embedding_without_pad(self.word_embedding, ids)

    def _forward_single(self, text: torch.Tensor = None, text_pad: torch.Tensor = None, text_num: torch.Tensor = None,
                        text_numpad: torch.Tensor = None, equation: torch.Tensor = None,
                        **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where B = batch size, T = length of input sequence, and H = dimension of input embedding.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :param torch.Tensor text_num:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H], where N = maximum number of written numbers in the batch.
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T], where B = batch size and T = length of expression sequence.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - 'op': Log probability of next op tokens (i.e. the 2nd equation in Appendix without argmax).
                FloatTensor with shape [B, T, V], where V = size of Op vocabulary.
        """
        # Retrieve decoder's hidden states
        # Dictionary will have '_out'
        result = super()._forward_single(text, text_pad, text_num, equation)

        # Take and pop internal states from the result dict.
        # Decoder's hidden states: [B, T, H]
        output = result.pop('_out')

        # Predict the next op token: Shape [B, T, V].
        op_out = self.op_out(output)
        result['op'] = self.softmax(op_out)

        return result

    def _build_target_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
            - 'op': Index of next op tokens. LongTensor with shape [B, T].
        """
        # Build targets
        return {'op': kwargs[IN_EQN]}


# Define the mapping between abbreviated name and submodule classes
SUBMODULE_TYPES = {
    MODEL_VANILLA_TRANS: VanillaOpTransformer,
    MODEL_EXPR_TRANS: ExpressionTransformer,
    MODEL_EXPR_PTR_TRANS: ExpressionPointerTransformer
}


__all__ = ['DecoderModel',
           'ExpressionPointerTransformer', 'ExpressionTransformer', 'VanillaOpTransformer', 'SUBMODULE_TYPES']
