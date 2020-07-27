from typing import Dict

import torch
from torch import nn

from page.config import ModelConfig
from page.const import *
from page.util import filter_dict_by_keys
from .equation import *
from .text import TextModel


class Solver(nn.Module):
    """
    Model that combines text encoder and expression/op decoder.
    """

    def __init__(self, text_model: TextModel, eqn_model: DecoderModel):
        """
        Initialize solver module

        :param TextModel text_model: Encoder model for reading problem text
        :param DecoderModel eqn_model: Decoder model for generating expression/op tokens
        """
        super().__init__()

        # Set sub-model instances.
        self.text_model = text_model
        self.eqn_model = eqn_model

    @property
    def required_field(self):
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return self.eqn_model.required_field

    @property
    def is_expression_type(self):
        """
        :rtype: bool
        :return: True if this model requires Expression type sequence
        """
        return self.eqn_model.is_expression_type

    def save_pretrained(self, save_directory: str):
        """
        Save current state of Solver Model.

        :param str save_directory: String that represents path to the directory where this will be saved.
        """

        # Save text model
        self.text_model.save_pretrained(save_directory)
        # Save equation model
        self.eqn_model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, config: ModelConfig):
        """
        Load pre-trained model

        :param ModelConfig config: Configuration of a model that will be loaded
        :rtype: Solver
        :return: A Solver instance
        """
        # Load text model
        text_model = TextModel.from_pretrained(config)
        # Load equation model
        eqn_model = SUBMODULE_TYPES[config.model_type].from_pretrained(config)

        # Return solver instance.
        return Solver(text_model, eqn_model)

    def forward(self, **kwargs):
        """
        Do forward pass.
        .. see::
            _forward_training or _forward_testing for further detail.
        """
        if self.training:
            # Run training if this is in training phase.
            return self._forward_training(**kwargs)
        else:
            # Otherwise, compute next tokens without computing gradients.
            with torch.no_grad():
                return self._forward_testing(**kwargs)

    def _forward_training(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward computation in training phase (using teacher-forcing)

        :keyword int max_numbers: Maximum number of numbers in the text. 1 by default.
        :keyword torch.Tensor text:
            Long Tensor representing text indices.
            Shape [B, S], where B = batch size, S = length of text
        :keyword torch.Tensor text_pad:
            Bool Tensor representing the position of padding in the text. Shape [B, S].
        :keyword torch.Tensor text_num:
            Bool Tensor representing the position of numbers in the text. Shape [B, S].
        :param torch.Tensor equation:
            Long Tensor containing expression/op tokens (This corresponds to f_i, a_ij or t_i in the paper)
            If the decoder use expression tokens:
                Shape: [B, T, 1+2A], where T = length of expression sequence and A = maximum arity.
            If the decoder use op tokens:
                Shape: [B, T, 2], where T = length of expression sequence.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of loss and accuracy information
            - Train_<key>/loss : loss of training to predict <key> tensors (teacher-forcing)
            - Train_<key>/acc_token : token-level accuracy when predicting <key> tensors (teacher-forcing)
            - Train_<key>/acc_seq : sequence-level accuracy when predicting <key> tensors (teacher-forcing)
            <key> is one of the following: operator, operand_0, operand_1, ..., op
        """
        # Encode the problem texts
        text = self.text_model(**kwargs)

        # Pass the decoder
        result = self.eqn_model(**text, equation=kwargs[IN_EQN])

        # Remove output keys except loss & accuracy.
        keys = [key for key in result if key.startswith('Train_')]
        result = filter_dict_by_keys(result, *keys)

        # Sum up all losses
        result['total_loss'] = 0
        for key, value in result.items():
            if key.endswith('/loss'):
                result['total_loss'] += value

        return result

    def _forward_testing(self, beam: int = 3, max_len: int = 128, function_arities: Dict[int, int] = None,
                         **kwargs) -> torch.Tensor:
        """
        Forward computation in evaluation phase (auto-regressive)

        :param int beam:
            Size of beams for beam search. 3 by default.
        :param int max_len:
            Size of maximum length for generation. We specified this to ensure termination of generation.
            128 by default. (This is much higher than the average number of tokens in the target equation sequence)
        :param Dict[int,int] function_arities:
            Mapping from operator index to its arity values. None by default (set all operator have the maximum arity).
        :keyword int max_numbers:
            Maximum number of numbers in the text. 1 by default.
        :keyword torch.Tensor text:
            Long Tensor representing text indices.
            Shape [B, S], where B = batch size, S = length of text
        :keyword torch.Tensor text_pad:
            Bool Tensor representing the position of padding in the text. Shape [B, S].
        :keyword torch.Tensor text_num:
            Bool Tensor representing the position of numbers in the text. Shape [B, S].
        :rtype: torch.Tensor
        :return:
            Long Tensor containing expression/op tokens (This corresponds to f_i, a_ij or t_i in the paper)
            If the decoder use expression tokens:
                Shape: [B, M, T, 1+2A],
                where M = size of beams, T = length of expression sequence, and A = maximum arity.
            If the decoder use op tokens:
                Shape: [B, M, T, 2], where T = length of expression sequence.
        """
        # Encode the problem texts, update keyword arguments and apply the decoder.
        text = self.text_model(**kwargs)

        # Proceed generation
        if self.is_expression_type:
            result = self._generate_expressions(text, max_len=max_len, beam=beam, function_arities=function_arities)
        else:
            result = self._generate_op(text, max_len=max_len, beam=beam)

        # To ensure having the same size between different data-parallel executions, pad it to the maximum length.
        shape = list(result.shape)
        seq_len = shape[2]
        if seq_len < max_len:
            shape[2] = max_len
            tensor = torch.full(shape, fill_value=PAD_ID, dtype=torch.long)
            tensor[:, :, :seq_len] = result.cpu()
            result = tensor

        return result.to(text[IN_TXT].device)

    def _generate_expressions(self, text: Dict[str, torch.Tensor], max_len=128, beam=3, function_arities=None):
        """
        Generate expression tokens

        :param Dict[str,torch.Tensor] text:
            Dictionary that contains encoder's hidden state and various information generated by the encoder
        :param int max_len:
            Size of maximum length for generation. We specified this to ensure termination of generation.
            128 by default. (This is much higher than the average number of tokens in the target equation sequence)
        :param int beam:
            Size of beams for beam search. 3 by default.
        :param Dict[int,int] function_arities:
            Mapping from operator index to its arity values. None by default (set all operator have the maximum arity).
        :rtype: torch.Tensor
        :return:
            Long Tensor representing op tokens. Shape [B, M, T, 1+2A],
            where B = batch size, M = size of beams, T = length of expression sequence, and A = maximum arity.
        """
        # Retrieve size & device information
        batch_sz = text[IN_TXT].shape[0]
        batch_range = range(batch_sz)
        device = text[IN_TXT].device
        arity = self.eqn_model.max_arity

        # Prepare range of numbers and constants
        if self.required_field == FIELD_EXPR_GEN:
            # The range of numbers and constants are static in this case.
            num_range = lambda n: 1 <= n < 1 + NUM_MAX
            con_range = lambda n: n == 0 or 1 + NUM_MAX + MEM_MAX <= n
            # This type treats all operands as constants (no offsets)
            num_offset = mem_offset = con_offset = 0
        else:
            # This type dynamically concatenates all source of operands.
            con_offset = 0
            num_offset = self.eqn_model.constant_vocab_size
            mem_offset = num_offset + text[IN_TNUM].shape[1]

            con_range = lambda n: n < num_offset
            num_range = lambda n: num_offset <= n < mem_offset

        # Map from operator to its arity
        function_arities = {} if function_arities is None else function_arities

        # Prepare inputs.
        # At the beginning, we start with only one beam, [B, M=1, T=1, 1+2A].
        init = [FUN_NEW_EQN_ID] + [PAD_ID] * (2 * arity)
        result = torch.tensor([[[init]] for _ in batch_range], dtype=torch.long)

        # Prepare storage for beam scores. [B, M=1]
        beamscores = torch.zeros(batch_sz, 1)

        # Prepare indicator for termination
        all_exit = False
        seq_len = 1

        while seq_len < max_len and not all_exit:
            # Compute scores for operator/operands
            scores = self.eqn_model(**text, equation=result.to(device))
            # Retrieve score of the last token. [B, M, T, ?] -> [B, M, ?]
            scores = {key: score[:, :, -1].cpu().detach() for key, score in scores.items()}

            # Probability score for each beam & function words. [B, M, V] + [B, M, 1] = [B, M, V]
            beam_function_score = scores['operator'] + beamscores.unsqueeze(-1)

            # Prepare storage for the next results
            next_beamscores = torch.zeros(batch_sz, beam)
            next_result = torch.full((batch_sz, beam, seq_len + 1, 1 + 2 * arity), fill_value=PAD_ID, dtype=torch.long)

            beam_range = range(beam_function_score.shape[1])
            operator_range = range(beam_function_score.shape[2])
            for i in batch_range:
                # Compute scores for all (Beam, Operator, Operand) combinations. We will add all log probabilities.
                # Storage for i-th item in the batch
                score_i = []
                for m in beam_range:
                    # For each beam, compute scores
                    # Check whether this beam was terminated before this step.
                    last_item = result[i, m, -1, 0].item()
                    after_last = last_item in {PAD_ID, FUN_END_EQN_ID}

                    if after_last:
                        # Score should be unchanged after __END_EQN token.
                        score_i.append((beamscores[i, m].item(), m, PAD_ID, []))
                        continue

                    # Compute beams for operators first.
                    operator_scores = {}
                    for f in operator_range:
                        operator_score = beam_function_score[i, m, f].item()

                        if f < len(FUN_TOKENS):
                            if f == FUN_END_EQN_ID and last_item == FUN_NEW_EQN_ID:
                                # Don't permit sequence like [__NEW_EQN, __END_EQN]
                                continue

                            # __NEW_EQN, __END_EQN, __NEW_VAR token does not require any arguments.
                            score_i.append((operator_score, m, f, []))
                        else:
                            operator_scores[f] = operator_score

                    # Combine operand log-probabilities with operator word log-probability.
                    operand_beams = [(0.0, [])]
                    for a in range(arity):
                        # Get top-k result
                        score_ia, index_ia = scores['operand_%s' % a][i, m].topk(beam)
                        score_ia = score_ia.tolist()
                        index_ia = index_ia.tolist()

                        # Compute M*M combination and preserve only top-M results.
                        operand_beams = [(s_prev + s_a, arg_prev + [arg_a])
                                         for s_prev, arg_prev in operand_beams
                                         for s_a, arg_a in zip(score_ia, index_ia)]
                        operand_beams = sorted(operand_beams, key=lambda t: t[0], reverse=True)[:beam]

                        for f, s_f in operator_scores.items():
                            # Append expression (pair of operator and operands) that match current arity.
                            if function_arities.get(f, arity) == a + 1:
                                score_i += [(s_f + s_args, m, f, args) for s_args, args in operand_beams]

                # Prepare the next beams. Scores[i] originally have shape [M, T] -> [M * T] after flattening.
                beam_registered = set()
                for score, prevbeam, operator, operands in sorted(score_i, key=lambda t: t[0], reverse=True):
                    if len(beam_registered) == beam:
                        # If beam was full, exit loop.
                        break

                    # Check whether this combination was already checked
                    beam_signature = (prevbeam, operator, *operands)
                    if beam_signature in beam_registered:
                        continue

                    # Set the next-beam
                    newbeam = len(beam_registered)
                    next_beamscores[i, newbeam] = score

                    # Copy tokens
                    next_result[i, newbeam, :-1] = result[i, prevbeam]
                    new_tokens = [operator]
                    for j, a in enumerate(operands):
                        # Assign operands and its source types.
                        if con_range(a):
                            new_tokens += [ARG_CON_ID, a - con_offset]
                        elif num_range(a):
                            new_tokens += [ARG_NUM_ID, a - num_offset]
                        else:
                            new_tokens += [ARG_MEM_ID, a - mem_offset]
                    new_tokens = torch.as_tensor(new_tokens, dtype=torch.long, device=device)
                    next_result[i, newbeam, -1, :new_tokens.shape[0]] = new_tokens

                    # Assign beam information
                    beam_registered.add(beam_signature)

            # Copy score information
            beamscores = next_beamscores

            # Update checks for termination
            last_tokens = next_result[:, :, -1, 0]
            all_exit = ((last_tokens == PAD_ID) | (last_tokens == FUN_END_EQN_ID)).all().item()

            result = next_result
            seq_len += 1

        return result

    def _generate_op(self, text: Dict[str, torch.Tensor], max_len=128, beam=3):
        """
        Generate op tokens

        :param Dict[str,torch.Tensor] text:
            Dictionary that contains encoder's hidden state and various information generated by the encoder
        :param int max_len:
            Size of maximum length for generation. We specified this to ensure termination of generation.
            128 by default. (This is much higher than the average number of tokens in the target equation sequence)
        :param int beam:
            Size of beams for beam search. 3 by default.
        :rtype: torch.Tensor
        :return:
            Long Tensor representing op tokens. Shape [B, M, T],
            where B = batch size, M = size of beams, and T = length of expression sequence
        """
        # Retrieve size & device information
        batch_sz = text[IN_TXT].shape[0]
        batch_range = range(batch_sz)
        device = text[IN_TXT].device

        # Prepare inputs.
        # At the beginning, we start with only one beam, [B, M=1, T=1].
        result = torch.tensor([[[SEQ_NEW_EQN_ID]] for _ in batch_range], dtype=torch.long)

        # Prepare storage for beam scores. [B, M=1]
        beamscores = torch.zeros(batch_sz, 1)

        # Prepare indicator for termination
        all_exit = False
        seq_len = 1

        while seq_len < max_len and not all_exit:
            # Compute scores
            # Retrieve score of the last token. [B, M, T, ?] -> [B, M, ?]
            scores = self.eqn_model(**text, equation=result.to(device))
            scores = scores['op'][:, :, -1].cpu().detach()

            # Probability score for each beam & token. [B, M, V] + [B, M, 1] = [B, M, V]
            beam_token_score = scores + beamscores.unsqueeze(-1)

            # Prepare storage for the next results
            next_beamscores = torch.zeros(batch_sz, beam)
            next_result = torch.full((batch_sz, beam, seq_len + 1), fill_value=PAD_ID, dtype=torch.long)

            beam_range = range(beam_token_score.shape[1])
            token_range = range(beam_token_score.shape[2])
            for i in batch_range:
                # Compute scores for all (Beam, OpToken) combinations. We will add all log probabilities.
                # Storage for i-th item in the batch
                score_i = []
                for m in beam_range:
                    # For each beam, compute scores
                    # Check whether this beam was terminated before this step.
                    last_item = result[i, m, -1].item()
                    after_last = last_item == PAD_ID or last_item == SEQ_END_EQN_ID

                    if after_last:
                        # Score should be unchanged after __END_EQN token.
                        score_i.append((beamscores[i, m].item(), m, PAD_ID))
                        continue

                    for v in token_range:
                        if v == SEQ_END_EQN_ID and last_item == SEQ_NEW_EQN_ID:
                            # Don't permit sequence like [__NEW_EQN, __END_EQN]
                            continue

                        token_score = beam_token_score[i, m, v].item()
                        score_i.append((token_score, m, v))

                # Prepare the next beams. Scores[i] originally have shape [M, T] -> [M * T] after flattening.
                beam_registered = set()
                for score, prevbeam, token in sorted(score_i, key=lambda t: t[0], reverse=True):
                    if len(beam_registered) == beam:
                        # If beam was full, exit loop.
                        break

                    if (prevbeam, token, token) in beam_registered:
                        # If this combination was already checked, do not add this.
                        continue

                    # Set the next-beam
                    newbeam = len(beam_registered)
                    next_beamscores[i, newbeam] = score

                    # Copy tokens
                    next_result[i, newbeam, :-1] = result[i, prevbeam]
                    next_result[i, newbeam, -1] = token

                    # Assign beam information
                    beam_registered.add((prevbeam, token, token))

            # Copy score information
            beamscores = next_beamscores

            # Update checks for termination
            last_token_ids = next_result[:, :, -1]
            all_exit = ((last_token_ids == PAD_ID) | (last_token_ids == SEQ_END_EQN_ID)).all().item()

            result = next_result
            seq_len += 1

        return result


__all__ = ['Solver']
