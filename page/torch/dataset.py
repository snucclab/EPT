import json
from collections import namedtuple
from pathlib import Path
from typing import List, Dict, Tuple

from torch import Tensor, load as load_data, save as save_data
from torchtext.data import batch
from torchtext.data.utils import RandomShuffler

from page.const import FUN_NEW_VAR
from .eq_field import OpEquationField, ExpressionEquationField
from .text_field import ProblemTextField

"""
Named tuple for representing a pair of input (problem text) and output (equation).
- 'text': Slot for problem text
- 'op_gen': Slot for output equation with op-token (Vanilla Transformer)
- 'expr_gen': Slot for output equation with expression-token (Vanilla Transformer + Expression)
- 'expr_ptr': Slot for output equation with expression-token and pointer (EPT)
- 'index': Slot for Problem ID in the dataset
- 'expected': Slot for Expected answer tuple 
"""
ProblemInstance = namedtuple('ProblemInstance', ('text', 'op_gen', 'expr_gen', 'expr_ptr', 'index', 'expected'))


def _get_token_length(item: ProblemInstance) -> Tuple[int, ...]:
    """
    Get length of input/output tokens to form a token-based batch.

    :param ProblemInstance item: ProblemInstance after preprocessed but before numericalized.
    :return: Tuple of lengths
        - [0] Length of text sequence in tokens
        - [1] Length of op-token sequence
        - [2] Length of expression-token sequence
    """
    # Note: expr_ptr and expr_gen has the same length
    # Note: we need to add 2 tokens to reserve space for BOS/EOS tokens
    return len(item.text.token) + 2, len(item.op_gen) + 2, len(item.expr_gen) + 2


def _get_item_size(item: ProblemInstance, size: int, prev_size: int) -> int:
    """
    Compute the size of new batch when adding an item into a batch.

    :param ProblemInstance item: Item to be added
    :param int size: Count of items that currently added in the batch
    :param int prev_size: Current size of the batch in terms of tokens.
    :return: New size of the batch in terms of tokens.
    """
    # Note: this function is called after adding item into the batch (so, `size` is already increased by 1)
    # Compute previous maximum length of tokens
    prev_max = prev_size // (size - 1) if size > 1 else 0
    # Compute new maximum length of tokens
    new_max = max(prev_max, *_get_token_length(item))
    # Return the new size of batch.
    return new_max * size


class TokenBatchIterator(object):
    """
    Batch iterator
    """

    def __init__(self, dataset: str, problem_field: ProblemTextField, op_gen_field: OpEquationField,
                 expr_gen_field: ExpressionEquationField, expr_ptr_field: ExpressionEquationField,
                 token_batch_size: int = 4096, testing_purpose: bool = False):
        """
        Instantiate batch iterator

        :param str dataset: Path of JSON with lines file to be loaded.
        :param ProblemTextField problem_field: Text field for encoder
        :param OpEquationField op_gen_field: OP-token equation field for decoder
        :param ExpressionEquationField expr_gen_field: Expression-token equation field for decoder (no pointer)
        :param ExpressionEquationField expr_ptr_field: Expression-token equation field for decoder (pointer)
        :param int token_batch_size: Maximum bound for batch size in terms of tokens.
        :param bool testing_purpose:
            True if this dataset is for testing. Otherwise, we will randomly shuffle the dataset.
        """
        # Define fields
        self.problem_field = problem_field
        self.op_gen_field = op_gen_field
        self.expr_gen_field = expr_gen_field
        self.expr_ptr_field = expr_ptr_field

        # Store the batch size
        self._batch_size = token_batch_size
        # Store whether this dataset is for testing or not.
        self._testing_purpose = testing_purpose

        # Storage for list of shuffled batches
        self._batches = None
        # Iterator for batches
        self._iterator = None
        # Random shuffler
        self._random = RandomShuffler() if not testing_purpose else None

        # Read the dataset.
        cached_path = Path(dataset + '.cached')
        if cached_path.exists():
            # If cached version is available, load the dataset from it.
            cache = load_data(cached_path)
            self._dataset = cache['dataset']
            vocab_cache = cache['vocab']

            # Restore vocabulary from the dataset.
            if self.op_gen_field.has_empty_vocab:
                self.op_gen_field.token_vocab = vocab_cache['token']
            if self.expr_gen_field.has_empty_vocab:
                self.expr_gen_field.operator_word_vocab = vocab_cache['func']
                self.expr_gen_field.constant_word_vocab = vocab_cache['arg']
            if self.expr_ptr_field.has_empty_vocab:
                self.expr_ptr_field.operator_word_vocab = vocab_cache['func']
                self.expr_ptr_field.constant_word_vocab = vocab_cache['const']
        else:
            # Otherwise, compute preprocessed result and cache it in the disk
            # First, read the JSON with lines file.
            _dataset = []
            _items_for_vocab = []
            with Path(dataset).open('r+t', encoding='UTF-8') as fp:
                for line in fp.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)

                    # We only need 'text', 'expr', 'id', and 'answer'
                    _dataset.append((item['text'], item['expr'], item['id'], item['answer']))
                    # Separately gather equations to build vocab.
                    _items_for_vocab.append(item['expr'])

            # Build vocab if it is empty
            if self.op_gen_field.has_empty_vocab:
                self.op_gen_field.build_vocab(_items_for_vocab)
            if self.expr_gen_field.has_empty_vocab:
                self.expr_gen_field.build_vocab(_items_for_vocab)
            if self.expr_ptr_field.has_empty_vocab:
                self.expr_ptr_field.build_vocab(_items_for_vocab)

            # Run preprocessing
            self._dataset = [self._tokenize_equation(item) for item in _dataset]

            # Cache dataset and vocabulary.
            save_data({'dataset': self._dataset,
                       'vocab': {
                           'token': self.op_gen_field.token_vocab,
                           'func': self.expr_gen_field.operator_word_vocab,
                           'arg': self.expr_gen_field.constant_word_vocab,
                           'const': self.expr_ptr_field.constant_word_vocab,
                       }}, cached_path)

        # Compute the number of examples
        self._examples = len(self._dataset)
        # Generate the batches.
        self.reset()

    def get_rng_state(self):
        """
        :return: The state of RNG used in this iterator.
        """
        return self._random.random_state

    def set_rng_state(self, state):
        """
        Restore the RNG state

        :param state: state of RNG to be restored
        """
        self._random = RandomShuffler(state)
        # Generate the batches by following the state
        self.reset()

    def print_item_statistics(self, logger):
        """
        Print the statistics of this dataset

        :param logger: Logger instance where this method writes log in
        """
        # Compute item statistics
        item_stats = self.get_item_statistics()

        # Record length information about text tokens
        lengths = item_stats['text_token']
        logger.info('Information about lengths of text sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / self._examples)

        # Record length information about op-tokens
        lengths = item_stats['eqn_op_token']
        logger.info('Information about lengths of token unit sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / self._examples)

        # Record vocabulary used in op-tokens
        logger.info('Token unit vocabulary (no-pointer): %s', self.op_gen_field.token_vocab.itos)

        # Record length information about expression-tokens
        lengths = item_stats['eqn_expr_token']
        logger.info('Information about lengths of operator unit sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / self._examples)

        # Record vocabulary used in expression-tokens
        logger.info('Operator unit vocabulary (operator): %s', self.expr_gen_field.operator_word_vocab.itos)
        logger.info('Operator unit vocabulary (operand): %s', self.expr_gen_field.constant_word_vocab.itos)
        logger.info('Operator unit vocabulary (constant): %s', self.expr_ptr_field.constant_word_vocab.itos)

    def get_item_statistics(self) -> Dict[str, List[int]]:
        """
        :rtype: Dict[str, List[int]]
        :return: Dictionary of dataset statistics per item.
        """
        return dict(
            # Text tokens per each item
            text_token=[len(item.text.token) for item in self._dataset],
            # Written numbers per each item
            text_number=[len(item.text.number_value) for item in self._dataset],
            # Op-tokens per each item
            eqn_op_token=[len(item.op_gen) for item in self._dataset],
            # Expression-tokens per each item
            eqn_expr_token=[len(item.expr_gen) for item in self._dataset],
            # Number of unknowns(variables) per each item
            eqn_unk=[sum(func == FUN_NEW_VAR for func, _ in item.expr_gen) for item in self._dataset]
        )

    def _tokenize_equation(self, item) -> ProblemInstance:
        """
        Tokenize the given equation

        :param item: Quadraple of (problem text, equation, problem ID, expected answer)
        :rtype: ProblemInstance
        :return: A ProblemInstance
            - text: ProblemTextInstance for given problem text
                - token: List of tokens in the text (with [NUM] token)
                - pad: None
                - number: None
                - number_value: Dictionary representing value of the numbers in the text.
            - op_gen: List of Op-tokens for given equation
            - expr_gen: List of Expression-tokens for given equation without pointing
            - expr_ptr: List of Expression-tokens for given equation with pointing
            - index: Problem ID in the dataset
            - expected: Expected Answer Tuple.
        """
        return ProblemInstance(
            text=self.problem_field.preprocess(item[0]),
            op_gen=self.op_gen_field.preprocess(item[1]),
            expr_gen=self.expr_gen_field.preprocess(item[1]),
            expr_ptr=self.expr_ptr_field.preprocess(item[1]),
            index=item[2],
            expected=item[3]
        )

    def reset(self):
        """
        (Re-)generate the batches and shuffle the order of them.
        """
        self._batches = list(self._generate_batches())

        if not self._testing_purpose:
            self._iterator = iter(self._random(self._batches))
        else:
            # Preserve the order when testing.
            self._iterator = iter(self._batches)

    def _generate_batches(self):
        """
        Make a generator for the batches.
        This method will enforce a batch have items with similar lengths.

        :return: This function yields a batched item (ProblemInstance)
            - text: ProblemTextInstance for given problem text
                - token: Long Tensor of index of text tokens. Shape [B, S],
                    where B = batch size and S = length of tokenized problem text sequence
                - pad: Bool Tensor for indicating padded positions, Shape [B, S].
                - number: Long Tensor for indicating number indices that a token belongs to. Shape [B, S].
                - number_value: Dictionary representing value of the numbers in the text.
            - op_gen: A LongTensor representing op-token indices. Shape [B, P],
                where P = length of op-token sequence.
            - expr_gen: A LongTensor representing expression-token indices (without pointer). Shape [B, X, 1+2A],
                where X = length of op-token sequence, and A = maximum arity.
            - expr_ptr: A LongTensor representing expression-token indices (with pointer). Shape [B, X, 1+2A]
            - index: List of problem IDs in the dataset
            - expected: List of expected answer tuples
        """
        max_token_size = 0
        items = []
        dataset = self._dataset

        # Chunk the dataset with much larger group of items than specified batch size.
        chunks = list(batch(dataset, self._batch_size * 1024, _get_item_size))
        for batch_group in chunks:
            # Sort within each group of items
            for item in sorted(batch_group, key=_get_token_length):
                items.append(item)

                # Compute the max-length key and new batch size.
                token_size = max(_get_token_length(item))
                max_token_size = max(max_token_size, token_size)
                batch_size = max_token_size * len(items)

                # If the size exceeded, flush it.
                if batch_size == self._batch_size:
                    yield self._concatenate_batch(items)
                    items = []
                    max_token_size = 0
                elif batch_size > self._batch_size:
                    yield self._concatenate_batch(items[:-1])
                    items = items[-1:]
                    max_token_size = token_size

            # If items is not empty, flush the last chunk.
            if items:
                yield self._concatenate_batch(items)

    def _concatenate_batch(self, items: List[ProblemInstance]) -> ProblemInstance:
        """
        Concatenate, pad & numericalize the batched items
        :param List[ProblemInstance] items: Items for a single batch
        :rtype: ProblemInstance
        :return: A batched item
            - text: ProblemTextInstance for given problem text
                - token: Long Tensor of index of text tokens. Shape [B, S],
                    where B = batch size and S = length of tokenized problem text sequence
                - pad: Bool Tensor for indicating padded positions, Shape [B, S].
                - number: Long Tensor for indicating number indices that a token belongs to. Shape [B, S].
                - number_value: Dictionary representing value of the numbers in the text.
            - op_gen: A LongTensor representing op-token indices. Shape [B, P],
                where P = length of op-token sequence.
            - expr_gen: A LongTensor representing expression-token indices (without pointer). Shape [B, X, 1+2A],
                where X = length of op-token sequence, and A = maximum arity.
            - expr_ptr: A LongTensor representing expression-token indices (with pointer). Shape [B, X, 1+2A]
            - index: List of problem IDs in the dataset
            - expected: List of expected answer tuples
        """
        kwargs = {}
        for item in items:
            for key in ProblemInstance._fields:
                if key not in kwargs:
                    kwargs[key] = []

                kwargs[key].append(getattr(item, key))

        # Pad and numericalize each I/O field.
        kwargs['text'] = self.problem_field.process(kwargs['text'])
        kwargs['op_gen'] = self.op_gen_field.process(kwargs['op_gen'])
        kwargs['expr_gen'] = self.expr_gen_field.process(kwargs['expr_gen'])
        kwargs['expr_ptr'] = self.expr_ptr_field.process(kwargs['expr_ptr'])

        return ProblemInstance(**kwargs)

    def __len__(self):
        """
        :rtype: int
        :return: Length of the iterator, i.e. the number of batches.
        """
        return len(self._batches)

    def __iter__(self):
        """
        :return: Iterator of batches
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        :return: The next batch. If this is not for testing purposes, batch will be infinitely generated.
        """
        try:
            return next(self._iterator)
        except StopIteration as e:
            if not self._testing_purpose:
                # Re-initialize iterator when iterator is empty
                self.reset()
                return self.__next__()
            else:
                raise e


__all__ = ['TokenBatchIterator', 'ProblemInstance']
