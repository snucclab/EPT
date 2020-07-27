from collections import Counter
from typing import List, Union, Tuple, Any

import regex as re
import torch
from torchtext.data import RawField
from torchtext.vocab import Vocab

from page.const import *


def postfix_parser(equation: List[Union[str, Tuple[str, Any]]], memory: list) -> int:
    """
    Read Op-token postfix equation and transform it into Expression-token sequence.

    :param List[Union[str,Tuple[str,Any]]] equation:
        List of op-tokens to be parsed into a Expression-token sequence
        Item of this list should be either
        - an operator string
        - a tuple of (operand source, operand value)
    :param list memory:
        List where previous execution results of expressions are stored
    :rtype: int
    :return:
        Size of stack after processing. Value 1 means parsing was done without any free expression.
    """
    stack = []

    for tok in equation:
        if tok in OPERATORS:
            # If token is an operator, form expression and push it into the memory and stack.
            op = OPERATORS[tok]
            arity = op['arity']

            # Retrieve arguments
            args = stack[-arity:]
            stack = stack[:-arity]

            # Store the result with index where the expression stored
            stack.append((ARG_MEM, len(memory)))
            # Store the expression into the memory.
            memory.append((tok, args))
        else:
            # Push an operand before meet an operator
            stack.append(tok)

    return len(stack)


class OpEquationField(RawField):
    """
    Equation Field that uses Op-token
    """

    def __init__(self, variable_prefixes, number_prefixes, constant_prefix):
        """
        Initialize Op-token Equation Field

        :param List[str] variable_prefixes: List of strings which are prefixes to represent variables
        :param List[str] number_prefixes: List of strings which are prefixes to represent numbers
        :param str constant_prefix: Prefix string to represent constants
        """
        super().__init__()

        # Store prefix information
        self.variable_prefixes = set(variable_prefixes)
        self.number_perfixes = set(number_prefixes)
        self.constant_prefix = constant_prefix

        # Prepare name for Op-token vocabulary.
        self.token_vocab = None

    @property
    def has_empty_vocab(self) -> bool:
        """
        :rtype: bool
        :return: True if the vocabulary is empty.
        """
        return self.token_vocab is None

    def preprocess(self, formulae: List[Tuple[int, str]]) -> List[str]:
        """
        Tokenize equation using Op tokens.

        :param List[Tuple[int,str]] formulae:
            List of equations. Each equation is a tuple of following.
            - [0] Indicates type of equation (0: equation, 1: answer tuple, and 2: memory)
            - [1] String of expression
        :rtype: List[str]
        :return: List of Op tokens.
        """
        assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                       "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

        tokens = []
        memory_counter = 0
        variables = {}

        for typ, expr in formulae:
            if type(expr) is str:
                expr = re.split('\\s+', expr.strip())

            if typ == PREP_KEY_ANS:
                # Ignore answer tuple
                continue
            elif typ == PREP_KEY_MEM:
                # If this is a memory, then make it as M_<id> = <expr>.
                expr = ['M_%s' % memory_counter] + expr + ['=']
                memory_counter += 1

            for token in expr:
                # Normalize tokens
                if any(token.startswith(prefix) for prefix in self.variable_prefixes):
                    # Remapping variables by order of appearance.
                    if token not in variables:
                        variables[token] = len(variables)

                    position = variables[token]
                    token = FORMAT_VAR % position  # By the index of the first appearance.
                    tokens.append(token)
                elif any(token.startswith(prefix) for prefix in self.number_perfixes):
                    # To preserve order, we padded indices with zeros at the front.
                    position = int(token.split('_')[-1])
                    tokens.append(FORMAT_NUM % position)
                else:
                    if token.startswith(self.constant_prefix):
                        token = token.replace(self.constant_prefix, CON_PREFIX)
                    tokens.append(token)

        return tokens

    def build_vocab(self, equations: List[List[Tuple[int, str]]]):
        """
        Build op token vocabulary.

        :param List[List[Tuple[int,str]]] equations: List of all equations in the dataset. (before preprocessing)
        """
        # Prepare occurrence counter
        equation_counter = Counter()

        for item in equations:
            # Build vocab for template or equation
            equation_counter.update([tok for tok in self.preprocess(item) if tok != PAD_ID])

        # Make sure that BOS and EOS always at the front of the vocabulary.
        special_tokens = SEQ_TOKENS.copy()

        # Enforce number and variable tokens are sorted with their indices.
        special_tokens += [FORMAT_NUM % i for i in range(NUM_MAX)]
        special_tokens += [FORMAT_VAR % i for i in range(VAR_MAX)]

        # Set the vocabulary.
        self.token_vocab = Vocab(equation_counter, specials=special_tokens)

    def process(self, batch: List[List[str]], device=None, **kwargs) -> torch.Tensor:
        """
        Generate Tensor representations for given problem texts

        :param List[List[str]] batch: List of preprocessed items to form a single batch
        :param torch.device device: Device to store
        :rtype: torch.Tensor
        :return: A LongTensor representing token indices. Shape [B, T],
            where B = batch size and T = length of op-token sequence
        """
        return self.numericalize(self.pad(batch), device=device)

    def pad(self, minibatch: List[List[str]]) -> List[List[int]]:
        """
        Pad minibatch to make each item have the same length

        :param List[List[str]] minibatch: List of pre-processed items to be padded
        :rtype: List[List[int]]
        :return: List of List, token indices with padded.
        """
        # Compute maximum length with __NEW_EQN and __END_EQN.
        max_len = max(len(item) for item in minibatch) + 2
        padded_batch = []

        for item in minibatch:
            # Convert item into IDs
            item = [self.token_vocab.stoi.get(tok, SEQ_UNK_TOK_ID) if tok != PAD_ID else tok
                    for tok in item]

            # Build padded item
            padded_item = [SEQ_NEW_EQN_ID] + item + [SEQ_END_EQN_ID]
            padded_item += [PAD_ID] * max(0, max_len - len(padded_item))

            padded_batch.append(padded_item)

        return padded_batch

    def numericalize(self, minibatch: List[List[int]], device=None) -> torch.Tensor:
        """
        Make a Tensor from the given minibatch.

        :param List[List[int]] minibatch: Padded minibatch to form a tensor
        :param torch.device device: Device to store the result
        :rtype: torch.Tensor
        :return: A Long Tensor of given minibatch. Shape [B, T],
            where B = batch size and T = length of op-token sequence
        """
        return torch.as_tensor(minibatch, dtype=torch.long, device=device)

    def convert_ids_to_equations(self, minibatch: torch.Tensor) -> List[List[str]]:
        """
        Convert minibatch to list of equations

        :param torch.Tensor minibatch:
            Long Tensor of index of text tokens. Shape [B, T],
            where B = batch size and T = length of op-token sequence
        :rtype: List[List[str]]
        :return: List of List of Op-tokens forming postfix equations
        """
        equation_batch = []
        for item in minibatch:
            equation = []

            # Tokens after PAD_ID will be ignored.
            for i, token in enumerate(item.tolist()):
                if token != PAD_ID:
                    token = self.token_vocab.itos[token]
                    if token == SEQ_NEW_EQN:
                        equation.clear()
                        continue
                    elif token == SEQ_END_EQN:
                        break
                else:
                    break

                equation.append(token)

            equation_batch.append(equation)
        return equation_batch


class ExpressionEquationField(RawField):
    """
    Equation field that uses Expression token
    """

    def __init__(self, variable_prefixes, number_prefixes, constant_prefix, max_arity: int = 2,
                 force_generation: bool = False):
        """
        Initialize Expression-token Equation Field

        :param List[str] variable_prefixes: List of strings which are prefixes to represent variables
        :param List[str] number_prefixes: List of strings which are prefixes to represent numbers
        :param str constant_prefix: Prefix string to represent constants
        :param int max_arity: Maximum arity of operator used in the dataset. 2 by default.
        :param bool force_generation: True if this field should use vocabulary instead of pointing. False by default.
        """
        super().__init__()

        # Store prefix information
        self.variable_prefixes = set(variable_prefixes)
        self.number_perfixes = set(number_prefixes)
        self.constant_prefix = constant_prefix

        # Store indicator of vocab / pointer
        self.force_generation = force_generation

        # Store maximum arity information
        self.max_arity = max_arity

        # Prepare name for Operator/Operand-token vocabulary
        self.operator_word_vocab = None
        self.constant_word_vocab = None

    @property
    def has_empty_vocab(self):
        """
        :rtype: bool
        :return: True if the vocabulary is empty.
        """
        return self.operator_word_vocab is None

    @property
    def function_arities(self):
        """
        :rtype: Dict[int, int]
        :return: Mapping from operator index to its arity values.
        """
        return {i: OPERATORS[f]['arity']
                for i, f in enumerate(self.operator_word_vocab.itos) if i >= len(FUN_TOKENS)}

    def preprocess(self, formulae: List[Tuple[int, str]]) -> List[Tuple[str, list]]:
        """
        Tokenize equation using Op tokens.

        :param List[Tuple[int,str]] formulae:
            List of equations. Each equation is a tuple of following.
            - [0] Indicates type of equation (0: equation, 1: answer tuple, and 2: memory)
            - [1] String of expression
        :rtype: List[Tuple[str, list]]
        :return: List of Expression tokens. Each expression is a tuple of following.
            - [0] Operator string
            - [1] List of operands. Each operand is tuple of following.
                - [0] Source of the operand. One of ARG_CON, ARG_NUM, ARG_MEM
                - [1] Value of the operand. String value for ARG_CON and integral value for the others.
        """
        assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                       "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

        variables = []
        memories = []

        for typ, expr in formulae:
            if type(expr) is str:
                expr = re.split('\\s+', expr.strip())

            # Replace number, const, variable tokens with N_<id>, C_<value>, X_<id>
            normalized = []
            for token in expr:
                if any(token.startswith(prefix) for prefix in self.variable_prefixes):
                    # Case 1: Variable
                    if token not in variables:
                        variables.append(token)

                    # Set as negative numbers, since we don't know how many variables are in the list.
                    normalized.append((ARG_MEM, - variables.index(token) - 1))
                elif any(token.startswith(prefix) for prefix in self.number_perfixes):
                    # Case 2: Number
                    token = int(token.split('_')[-1])
                    if self.force_generation:
                        # Treat number indicator as constant.
                        normalized.append((ARG_NUM, FORMAT_NUM % token))
                    else:
                        normalized.append((ARG_NUM, token))
                elif token.startswith(self.constant_prefix):
                    normalized.append((ARG_CON, token.replace(self.constant_prefix, CON_PREFIX)))
                else:
                    normalized.append(token)

            # Build expressions (ignore answer tuples)
            if typ == PREP_KEY_EQN:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Equation is not correct! '%s'" % expr
            elif typ == PREP_KEY_MEM:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Intermediate representation of memory is not correct! '%s'" % expr

        # Reconstruct indices for result of prior expression.
        var_length = len(variables)
        # Add __NEW_VAR at the front of the sequence. The number of __NEW_VAR()s equals to the number of variables used.
        preprocessed = [(FUN_NEW_VAR, []) for _ in range(var_length)]
        for operator, operands in memories:
            # For each expression
            new_arguments = []
            for typ, tok in operands:
                if typ == ARG_MEM:
                    # Shift index of prior expression by the number of variables.
                    tok = tok + var_length if tok >= 0 else -(tok + 1)

                    if self.force_generation:
                        # Build as a string
                        tok = FORMAT_MEM % tok

                new_arguments.append((typ, tok))

            # Register an expression
            preprocessed.append((operator, new_arguments))

        return preprocessed

    def build_vocab(self, equations: List[List[Tuple[int, str]]]):
        """
        Build op token vocabulary.

        :param List[List[Tuple[int,str]]] equations: List of all equations in the dataset. (before preprocessing)
        """
        # Prepare occurrence counter
        operator_counter = Counter()
        constant_counter = Counter()

        constant_specials = [ARG_UNK]
        if self.force_generation:
            # Enforce index of numbers become 1 ~ NUM_MAX
            constant_specials += [FORMAT_NUM % i for i in range(NUM_MAX)]
            # Enforce index of memory indices become NUM_MAX+1 ~ NUM_MAX+MEM_MAX
            constant_specials += [FORMAT_MEM % i for i in range(MEM_MAX)]

        for item in equations:
            # Equation is not tokenized
            item = self.preprocess(item)
            # Count operators
            operator, operands = zip(*item)
            operator_counter.update(operator)
            for operand in operands:
                # Count constant operands (all operands if self.force_generation)
                constant_counter.update([const for t, const in operand if t == ARG_CON or self.force_generation])

        # Set the vocabulary.
        self.operator_word_vocab = Vocab(operator_counter, specials=FUN_TOKENS_WITH_EQ.copy())
        self.constant_word_vocab = Vocab(constant_counter, specials=constant_specials)

    def process(self, batch: List[List[Tuple[str, list]]], device=None, **kwargs):
        """
        Generate Tensor representations for given problem texts

        :param List[List[Tuple[str,list]] batch: List of preprocessed items to form a single batch
        :param torch.device device: Device to store
        :rtype: torch.Tensor
        :return: A LongTensor representing token indices. Shape [B, T, 1+2A],
            where B = batch size, T = length of op-token sequence, and A = maximum arity.
        """
        return self.numericalize(self.pad(batch), device=device)

    def pad(self, minibatch: List[List[Tuple[str, list]]]) -> List[List[Tuple[str, list]]]:
        """
        Pad minibatch to make each item have the same length

        :param List[Tuple[str,list]] minibatch: List of pre-processed items to be padded
        :rtype: List[Tuple[str,list]]
        :return: List of List, token indices with padded.
        """
        # Compute maximum length with __NEW_EQN and __END_EQN.
        max_len = max(len(item) for item in minibatch) + 2  # 2 = BOE/EOE
        padded_batch = []

        # Padding for no-operand functions (i.e. special commands)
        max_arity_pad = [(None, None)] * self.max_arity

        for item in minibatch:
            padded_item = [(FUN_NEW_EQN, max_arity_pad)]

            for operator, operands in item:
                # We also had to pad operands.
                remain_arity = max(0, self.max_arity - len(operands))
                operands = operands + max_arity_pad[:remain_arity]

                padded_item.append((operator, operands))

            padded_item.append((FUN_END_EQN, max_arity_pad))
            padded_item += [(None, max_arity_pad)] * max(0, max_len - len(padded_item))

            # Add batched item
            padded_batch.append(padded_item)

        return padded_batch

    def convert_token_to_id(self, expression: Tuple[str, list]):
        """
        Convert an expression token to list of indices

        :param Tuple[str,list] expression: A tuple representing expression token
            - [0] Operator string
            - [1] List of operands. Each item is a tuple of the following.
                - [0] Source of the operand
                - [1] Value of the operand
        :return:
        """
        # Destructure the tuple.
        operator, operand = expression

        # Convert operator into index.
        operator = PAD_ID if operator is None else self.operator_word_vocab.stoi[operator]
        # Convert operands
        new_operands = []
        for src, a in operand:
            # For each operand, we attach [Src, Value] after the end of new_args.
            if src is None:
                new_operands += [PAD_ID, PAD_ID]
            else:
                # Get the source
                new_operands.append(ARG_TOKENS.index(src))
                # Get the index of value
                if src == ARG_CON or self.force_generation:
                    # If we need to look up the vocabulary, then find the index in it.
                    new_operands.append(self.constant_word_vocab.stoi.get(a, ARG_UNK_ID))
                else:
                    # Otherwise, use the index information that is already specified in the operand.
                    new_operands.append(a)

        # Return the flattened list of operator and operands.
        return [operator] + new_operands

    def numericalize(self, minibatch: List[List[Tuple[str, list]]], device=None) -> torch.Tensor:
        """
        Make a Tensor from the given minibatch.

        :param List[List[Tuple[str, list]]] minibatch: Padded minibatch to form a tensor
        :param torch.device device: Device to store the result
        :rtype: torch.Tensor
        :return: A Long Tensor of given minibatch. Shape [B, T, 1+2A],
            where B = batch size, T = length of op-token sequence, and A = maximum arity.
        """
        minibatch = [[self.convert_token_to_id(token) for token in item] for item in minibatch]
        return torch.as_tensor(minibatch, dtype=torch.long, device=device)

    def convert_ids_to_expressions(self, minibatch: torch.Tensor) -> List[List[Tuple[str, list]]]:
        """
        Convert minibatch to list of expression tokens.

        :param torch.Tensor minibatch:
            Long Tensor of index of text tokens. Shape [B, T, 1+2A],
            where B = batch size, T = length of op-token sequence, and A = maximum arity
        :rtype: List[List[Tuple[str, list]]]
        :return: List of List of expression tokens.
        """
        expression_batch = []

        for item in minibatch.tolist():
            # For each batch
            expressions = []

            for token in item:
                # For each token in the item.
                # First index should be the operator.
                operator = self.operator_word_vocab.itos[token[0]]
                if operator == FUN_NEW_EQN:
                    # If the operator is __NEW_EQN, we ignore the previously generated outputs.
                    expressions.clear()
                    continue

                if operator == FUN_END_EQN:
                    # If the operator is __END_EQN, we ignore the next outputs.
                    break

                # Now, retrieve the operands
                operands = []
                for i in range(1, len(token), 2):
                    # For each argument, we build two values: source and value.
                    src = token[i]
                    if src != PAD_ID:
                        # If source is not a padding, compute the value.
                        src = ARG_TOKENS[src]
                        operand = token[i + 1]
                        if src == ARG_CON or self.force_generation:
                            operand = self.constant_word_vocab.itos[operand]

                        if type(operand) is str and operand.startswith(MEM_PREFIX):
                            operands.append((ARG_MEM, int(operand[2:])))
                        else:
                            operands.append((src, operand))

                # Append an expression
                expressions.append((operator, operands))

            # Append an item
            expression_batch.append(expressions)

        return expression_batch

    def convert_ids_to_equations(self, minibatch: torch.Tensor) -> List[List[str]]:
        """
        Convert minibatch to list of equations

        :param torch.Tensor minibatch:
            Long Tensor of index of text tokens. Shape [B, T, 1+2A],
            where B = batch size, T = length of op-token sequence, and A = maximum arity
        :rtype: List[List[str]]
        :return: List of List of strings forming postfix equations
        """
        # Convert tensor to list of expressions
        expression_batch = self.convert_ids_to_expressions(minibatch)
        equation_batch = []

        for item in expression_batch:
            # For each batch.
            computation_history = []
            expression_used = []

            for operator, operands in item:
                # For each expression.
                computation = []

                if operator == FUN_NEW_VAR:
                    # Generate new variable whenever __NEW_VAR() appears.
                    computation.append(FORMAT_VAR % len(computation_history))
                else:
                    # Otherwise, form an expression tree
                    for src, operand in operands:
                        # Find each operands from specified sources.
                        if src == ARG_NUM and not self.force_generation:
                            # If this is a number pointer, then replace it into number indices
                            computation.append(FORMAT_NUM % operand)
                        elif src == ARG_MEM:
                            # If this indicates the result of prior expression, then replace it with prior results
                            if operand < len(computation_history):
                                computation += computation_history[operand]
                                # Mark the prior expression as used.
                                expression_used[operand] = True
                            else:
                                # Expression is not found, then use UNK.
                                computation.append(ARG_UNK)
                        else:
                            # Otherwise, this is a constant: append the operand itself.
                            computation.append(operand)

                    # To make it as a postfix representation, append operator at the last.
                    computation.append(operator)

                # Save current expression into the history.
                computation_history.append(computation)
                expression_used.append(False)

            # Find unused computation history. These are the top-level formula.
            computation_history = [equation for used, equation in zip(expression_used, computation_history) if not used]
            # Flatten the history and append it into the batch.
            equation_batch.append(sum(computation_history, []))

        return equation_batch


__all__ = ['OpEquationField', 'ExpressionEquationField']
