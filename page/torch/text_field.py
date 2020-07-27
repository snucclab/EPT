from collections import namedtuple
from typing import List

import torch
from torchtext.data import RawField
from transformers import PreTrainedTokenizer

from page.const import *
from page.util import find_numbers_in_text

"""
Text instance field for encoder
    - token: List of tokens in the text
    - pad: Indicating padded positions
    - number: Indictaing number positions or list of numbers in the text
    - number_value: Value of the numbers in the text.
"""
ProblemTextInstance = namedtuple('ProblemTextInstance', ('token', 'pad', 'number', 'number_value'))

# Special Underline used in SentencePiece
SPIECE_UNDERLINE = '▁'


class ProblemTextField(RawField):
    """
    Problem Text Field
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, maximum_sequence_length: int = 510):
        """
        Initialize Problem Text Field

        :param PreTrainedTokenizer tokenizer:
            Pre-trained tokenizer instance.
        :param int maximum_sequence_length:
            Maximum sequence length that can be handled in pre-trained model. (510 by default)
        """
        super().__init__()

        # Assign tokenizer
        self.tokenizer = tokenizer

        # Set size for transformation
        self.maximum_sequence_length = maximum_sequence_length

        # Add [NUM] token to mark the position of a number
        self.num_token_id = len(tokenizer) + 1
        self.pretrained_tokens = tokenizer.vocab_size
        self.tokenizer.add_special_tokens({'additional_special_tokens': [NUM_TOKEN]})

    def preprocess(self, item: str) -> ProblemTextInstance:
        """
        Find number and tokenize a problem text.

        :param str item: Text to be preprocessed
        :rtype: ProblemTextInstance
        :return: A named tuple of four items
            - token: List of tokens in the text (with [NUM] token)
            - pad: None
            - number: None
            - number_value: Dictionary representing value of the numbers in the text.
        """
        tokenized, numbers = find_numbers_in_text(item, append_number_token=True)
        tokenized = self.tokenizer.tokenize(tokenized.strip())

        return ProblemTextInstance(tokenized, None, None, numbers)

    def process(self, batch: List[ProblemTextInstance], device=None, **kwargs):
        """
        Generate Tensor representations for given problem texts

        :param List[ProblemTextInstance] batch: List of preprocessed items to form a single batch
        :param torch.device device: Device to store
        :rtype: ProblemTextInstance
        :return: A named tuple of four items
            - token: Long Tensor of index of text tokens. Shape [B, S],
                where B = batch size and S = length of tokenized problem text sequence
            - pad: Bool Tensor for indicating padded positions, Shape [B, S].
            - number: Long Tensor for indicating number indices that a token belongs to. Shape [B, S].
            - number_value: Dictionary representing value of the numbers in the text.
        """
        # First pad tokens, and then numericalize them.
        return self.numericalize(self.pad(batch), device=device)

    def pad(self, minibatch: List[ProblemTextInstance]) -> ProblemTextInstance:
        """
        Pad minibatch to make each item have the same length

        :param List[ProblemTextInstance] minibatch: List of pre-processed items to be padded
        :rtype: ProblemTextInstance
        :return: A named tuple of four mini-batched items
            - token: List[List[Int]] of index of text tokens. Shape [B, S],
                where B = batch size and S = length of tokenized problem text sequence
            - pad: None
            - number: List[List[Int]] for indicating number indices that a token belongs to. Shape [B, S].
            - number_value: Dictionary representing value of the numbers in the text.
        """
        # Compute maximum sequence length without BOS and EOS
        max_len = max(len(x.token) - x.count(NUM_TOKEN) for x in minibatch)
        if self.maximum_sequence_length:
            max_len = min(max_len, self.maximum_sequence_length)

        # Maximum sequence length with BOS and EOS
        max_len_with_specials = max_len + 2
        # Storage for padded values
        padded = []
        numbers = []
        num_pos = []

        # Shortcut for BOS, EOS, PAD token
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        pad_token = self.tokenizer.pad_token

        for item in minibatch:
            tokens = []
            number_indicators = []
            number_index = 0

            # We add tokens except [NUM], which we added to mark the position of numbers
            for tok in item.token:
                if tok != NUM_TOKEN:
                    # If this is not a [NUM] token, just add it.
                    tokens.append(tok)
                    # We don't know whether the token is representing a number or not yet, so set it as PAD
                    number_indicators.append(PAD_ID)
                else:
                    # If this is a [NUM] token, then previous tokens that form a single word are representing numbers.
                    # Set number index until we meet SPIECE_UNDERLINE (Beginning of a word).
                    for i in range(-1, -len(tokens) - 1, -1):
                        # From -1 to -len(tok) (check a token backward)
                        if tokens[i] != SPIECE_UNDERLINE:
                            # We ignore SPIECE_UNDERLINE token when marking the position of numbers.
                            # Note that this code does not ignore tokens starting with SPIECE_UNDERLINE.
                            number_indicators[i] = number_index

                        if tokens[i].startswith(SPIECE_UNDERLINE):
                            # Break when we meet the beginning of a word.
                            break

                    # Increase index of written numbers
                    number_index += 1

            # Check whether any number token is discarded.
            assert max(number_indicators[max_len:], default=PAD_ID) == PAD_ID, \
                "A number token should not be discarded. You should increase the number of input tokens."
            assert number_index == len(item.number_value) and len(set(number_indicators)) - 1 == number_index, \
                "The extracted numbers are not the same! %s vs %s" % (number_index, len(item.number_value))

            # Build tokens
            tokens = [bos_token] + tokens[:max_len] + [eos_token]
            number_indicators = [PAD_ID] + number_indicators[:max_len] + [PAD_ID]

            # Pad and append the item
            remain_len = max(0, max_len_with_specials - len(tokens))
            padded.append(tokens + [pad_token] * remain_len)
            num_pos.append(number_indicators + [PAD_ID] * remain_len)
            numbers.append(item.number_value)

        return ProblemTextInstance(padded, None, num_pos, numbers)

    def numericalize(self, minibatch: ProblemTextInstance, device=None):
        """
        Make a Tensor from the given minibatch.

        :param ProblemTextInstance minibatch: Padded minibatch to be numericalized
        :param torch.device device: Device to store the result
        :rtype: ProblemTextInstance
        :return: A named tuple of four items
            - token: Long Tensor of index of text tokens. Shape [B, S],
                where B = batch size and S = length of tokenized problem text sequence
            - pad: Bool Tensor for indicating padded positions, Shape [B, S].
            - number: Long Tensor for indicating number indices that a token belongs to. Shape [B, S].
            - number_value: Dictionary representing value of the numbers in the text.
        """
        # Convert tokens to token ids.
        tokens = [self.tokenizer.convert_tokens_to_ids(tok) for tok in minibatch.token]
        token_ids = torch.as_tensor(tokens, dtype=torch.long, device=device)
        # Padding mask: [True] if the position represents [PAD] token
        pad_masks = token_ids == self.tokenizer.pad_token_id
        # Number positions
        number_positions = torch.as_tensor(minibatch.number, dtype=torch.long, device=device)

        # LongTensor [B, S] of indices, and BoolTensor [B, S] indicates whether padding or not.
        return ProblemTextInstance(token_ids, pad_masks, number_positions, minibatch.number_value)

    def convert_ids_to_string(self, minibatch: torch.Tensor) -> List[str]:
        """
        Convert minibatch to list of problem texts

        :param torch.Tensor minibatch:
            Long Tensor of index of text tokens. Shape [B, S],
            where B = batch size and S = length of tokenized problem text sequence
        :rtype: List[str]
        :return: List of strings, each of which contains a problem text.
        """
        return [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(text.tolist()))
                for text in minibatch]


__all__ = ['ProblemTextField', 'ProblemTextInstance']
