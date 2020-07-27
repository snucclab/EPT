import re
from math import pi, e
from typing import Tuple, List, Union

from page.const import NUM_TOKEN

# Pattern of fraction numbers e.g. 5/3
FRACTIONAL_PATTERN = re.compile('(\\d+/\\d+)')
# Pattern of numbers e.g. 2,930.34
NUMBER_PATTERN = re.compile('([+\\-]?(\\d{1,3}(,\\d{3})+|\\d+)(\\.\\d+)?)')
# Pattern of number and fraction numbers
NUMBER_AND_FRACTION_PATTERN = re.compile('(%s|%s)' % (FRACTIONAL_PATTERN.pattern, NUMBER_PATTERN.pattern))
# Pattern of numbers that following zeros under the decimal point. e.g., 0_250000000
FOLLOWING_ZERO_PATTERN = re.compile('(\\d+|\\d+_[0-9]*[1-9])_?(0+|0{4}\\d+)$')

# Map specifies how english words can be interpreted as a number
NUMBER_READINGS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000,

    'once': 1, 'twice': 2, 'thrice': 3, 'double': 2, 'triple': 3, 'quadruple': 4,
    'doubled': 2, 'tripled': 3, 'quadrupled': 4,

    'third': 3, 'forth': 4, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13,
    'fourteenth': 14, 'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19,
    'twentieth': 20, 'thirtieth': 30, 'fortieth': 40, 'fiftieth': 50, 'sixtieth': 60,
    'seventieth': 70, 'eightieth': 80, 'ninetieth': 90,
    'hundredth': 100, 'thousandth': 1000, 'millionth': 1000000,
    'billionth': 1000000000,

    'dozen': 12, 'half': 0.5, 'quarter': 0.25,
    'halved': 0.5, 'quartered': 0.25,
}

# List of multiples, that can be used as a unit. e.g. three half (3/2)
MULTIPLES = ['once', 'twice', 'thrice', 'double', 'triple', 'quadruple', 'dozen', 'half', 'quarter',
             'doubled', 'tripled', 'quadrupled', 'halved', 'quartered']

# Suffix of plural forms
PLURAL_FORMS = [('ies', 'y'), ('ves', 'f'), ('s', '')]

# Precedence of operators
OPERATOR_PRECEDENCE = {
    '^': 4,
    '*': 3,
    '/': 3,
    '+': 2,
    '-': 2,
    '=': 1
}


def find_numbers_in_text(text: str, append_number_token: bool = False) -> Tuple[str, List[dict]]:
    """
    Find decimal, fractional or textual numbers written in the text.

    :param str text: String to be scanned
    :param bool append_number_token:
        True if this method need to append [NUM] directly after a number found.
    :return: A tuple of string and dictionary.
        - [0] String of scanned/processed text (If append_number_token is True, several [NUM] markers are added)
        - [1] List of dictionaries, each of which explains a number
            - 'token': Index of token (Index counted by the number space before)
            - 'value': Value of the token
            - 'is_text': True if this is a textual number, e.g. one
            - 'is_integer': True if this indicates an integral number
            - 'is_ordinal': True if this indicates an ordinal number, e.g. first
            - 'is_fraction': True if this indicates a fraction, e.g. 3/5 or three-fifths
            - 'is_single_multiple': True if this belongs to MULTIPLES
            - 'is_combined_multiple': True if this is a combination of a numeric token and MULTIPLES
    """
    numbers = []
    new_text = []

    # Replace "[NON-DIGIT][SPACEs].[DIGIT]" with "[NON-DIGIT][SPACEs]0.[DIGIT]"
    text = re.sub("([^\\d.,]+\\s*)(\\.\\d+)", "\\g<1>0\\g<2>", text)
    # Replace space between digits or digit and special characters like ',.' with "⌒" (to preserve original token id)
    text = re.sub("(\\d+)\\s+(\\.\\d+|,\\d{3}|\\d{3})", "\\1⌒\\2", text)

    # Original token index
    i = 0
    prev_token = None
    for token in text.split(' '):
        # Increase token id and record original token indices
        token_index = [i + j for j in range(token.count('⌒') + 1)]
        i = max(token_index) + 1

        # First, find the number patterns in the token
        token = token.replace('⌒', '')
        number_patterns = NUMBER_AND_FRACTION_PATTERN.findall(token)
        if number_patterns:
            for pattern in number_patterns:
                # Matched patterns, listed by order of occurrence.
                surface_form = pattern[0]
                surface_form = surface_form.replace(',', '')

                # Normalize the form: use the decimal point representation with 15-th position under the decimal point.
                is_fraction = '/' in surface_form
                value = eval(surface_form)
                if type(value) is float:
                    surface_form = FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=False, is_integer='.' not in surface_form,
                                    is_ordinal=False, is_fraction=is_fraction,
                                    is_single_multiple=False, is_combined_multiple=False))

            new_text.append(NUMBER_AND_FRACTION_PATTERN.sub(' \\1 %s ' % NUM_TOKEN, token))
        else:
            # If there is no numbers in the text, then find the textual numbers.
            # Append the token first.
            new_text.append(token)

            # Type indicator
            is_ordinal = False
            is_fraction = False
            is_single_multiple = False
            is_combined_multiple = False

            subtokens = re.split('[^a-zA-Z0-9]+', token.lower())  # Split hypen-concatnated tokens like twenty-three
            token_value = None
            for subtoken in subtokens:
                if not subtoken:
                    continue

                # convert to singular nouns
                for plural, singluar in PLURAL_FORMS:
                    if subtoken.endswith(plural):
                        subtoken = subtoken[:-len(plural)] + singluar
                        break

                if subtoken in NUMBER_READINGS:
                    if not token_value:
                        # If this is the first value in the token, then set it as it is.
                        token_value = NUMBER_READINGS[subtoken]

                        is_ordinal = subtoken[-2:] in ['rd', 'th']
                        is_single_multiple = subtoken in MULTIPLES

                        if is_ordinal and prev_token == 'a':
                            # Case like 'A third'
                            token_value = 1 / token_value
                    else:
                        # If a value was set before reading this subtoken,
                        # then treat it as multiples. (e.g. one-million, three-fifths, etc.)
                        followed_value = NUMBER_READINGS[subtoken]
                        is_single_multiple = False
                        is_ordinal = False

                        if followed_value >= 100 or subtoken == 'half':  # case of unit
                            token_value *= followed_value
                            is_combined_multiple = True
                        elif subtoken[-2:] in ['rd', 'th']:  # case of fractions
                            token_value /= followed_value
                            is_fraction = True
                        else:
                            token_value += followed_value

            # If a number is found.
            if token_value is not None:
                if type(token_value) is float:
                    surface_form = FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % token_value)
                else:
                    surface_form = str(token_value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=True, is_integer='.' not in surface_form,
                                    is_ordinal=is_ordinal, is_fraction=is_fraction,
                                    is_single_multiple=is_single_multiple, is_combined_multiple=is_combined_multiple))
                new_text.append(NUM_TOKEN)

        prev_token = token

    if append_number_token:
        text = ' '.join(new_text)

    return text, numbers


def constant_number(const: Union[str, int, float]) -> Tuple[bool, str]:
    """
    Converts number to constant symbol string (e.g. 'C_3').
    To avoid sympy's automatic simplification of operation over constants.

    :param Union[str,int,float,Expr] const: constant value to be converted.
    :return: (str) Constant symbol string represents given constant.
    """
    if type(const) is str:
        if const in ['C_pi', 'C_e', 'const_pi', 'const_e']:
            # Return pi, e as itself.
            return True, const.replace('const_', 'C_')

        # Otherwise, evaluate string and call this function with the evaluated number
        const = float(const.replace('C_', '').replace('const_', '').replace('_', '.'))
        return constant_number(const)
    elif type(const) is int or int(const) == float(const):
        # If the value is an integer, we trim the following zeros under decimal points.
        return const >= 0, 'C_%s' % int(abs(const))
    else:
        if abs(const - pi) < 1E-2:  # Including from 3.14
            return True, 'C_pi'
        if abs(const - e) < 1E-4:  # Including from 2.7182
            return True, 'C_e'

        # If the value is not an integer, we write it and trim followed zeros.
        # We need to use '%.15f' formatting because str() may gives string using precisions like 1.7E-3
        # Also we will trim after four zeros under the decimal like 0.05000000074 because of float's precision.
        return const >= 0, 'C_%s' % \
               FOLLOWING_ZERO_PATTERN.sub('\\1', ('%.15f' % abs(const)).replace('.', '_'))


def infix_to_postfix(equation: Union[str, List[str]], number_token_map: dict, free_symbols: list,
                     join_output: bool = True):
    """
    Read infix equation string and convert it into a postfix string

    :param Union[str,List[str]] equation:
        Either one of these.
        - A single string of infix equation. e.g. "5 + 4"
        - Tokenized sequence of infix equation. e.g. ["5", "+", "4"]
    :param dict number_token_map:
        Mapping from a number token to its anonymized representation (e.g. N_0)
    :param list free_symbols:
        List of free symbols (for return)
    :param bool join_output:
        True if the output need to be joined. Otherwise, this method will return the tokenized postfix sequence.
    :rtype: Union[str, List[str]]
    :return:
        Either one of these.
        - A single string of postfix equation. e.g. "5 4 +"
        - Tokenized sequence of postfix equation. e.g. ["5", "4", "+"]
    """
    # Template in ALG514/DRAW is already tokenized, without parenthesis.
    # Template in MAWPS is also tokenized but contains parenthesis.
    output_tokens = []
    postfix_stack = []

    # Tokenize the string if that is not tokenized yet.
    if type(equation) is str:
        equation = equation.strip().split(' ')

    # Read each token
    for tok in equation:
        # Ignore blank token
        if not tok:
            continue

        if tok == ')':
            # Pop until find the opening paren '('
            while postfix_stack:
                top = postfix_stack.pop()
                if top == '(':
                    # Discard the matching '('
                    break
                else:
                    output_tokens.append(top)
        elif tok in '*/+-=(':
            # '(' has the highest precedence when in the input string.
            precedence = OPERATOR_PRECEDENCE.get(tok, 1E9)

            while postfix_stack:
                # Pop until the top < current_precedence.
                # '(' has the lowest precedence in the stack.
                top = postfix_stack[-1]
                if OPERATOR_PRECEDENCE.get(top, -1E9) < precedence:
                    break
                else:
                    output_tokens.append(postfix_stack.pop())
            postfix_stack.append(tok)
        elif NUMBER_PATTERN.fullmatch(tok) is not None:
            # Just output the operand.
            positive, const_code = constant_number(tok)
            if not positive:
                const_code = const_code + '_NEG'
            output_tokens.append(const_code)
        elif tok in number_token_map:
            # Just output the operand
            output_tokens += number_token_map[tok]
        else:
            # This is a variable name
            # Just output the operand.
            if tok not in free_symbols:
                free_symbols.append(tok)

            tok = 'X_%s' % free_symbols.index(tok)
            output_tokens.append(tok)

    while postfix_stack:
        output_tokens.append(postfix_stack.pop())

    if join_output:
        return ' '.join(output_tokens)
    else:
        return output_tokens


def filter_dict_by_keys(dictionary: dict, *keys) -> dict:
    """
    Restrict dictionary onto given keys.

    :param dict dictionary: Dictionary to be restricted.
    :param str keys: Variable-length arguments for keys to be filtered
    :rtype: dict
    :return: restricted dictionary.
    """
    return {key: dictionary[key] for key in keys if key in dictionary}


class ExpectedTimeToFinishCalculator:
    """
    Class for computing expected time to finish
    """

    def __init__(self, total: int, current: int = 0):
        """
        Class for computing expected time to finish

        :param int total: Number of total items
        :param int current: Number of currently proceeded items
        """
        from datetime import datetime

        self.total = total - current
        self.current = 0

        # Store when this method called
        self.begin_time = datetime.now()
        self._time_delta = lambda: datetime.now() - self.begin_time

    def _eta(self):
        """
        Compute estimated time to finish

        :rtype: str
        :return: String representing the estimated time to finish
        """
        return (self.begin_time + self._time_delta() * (self.total / self.current)).strftime('%m-%d %H:%M:%S')

    def step(self, increase: int = 1):
        """
        Progress the counter and get the estimated time to finish.

        :param int increase: Number of items that are done.
        :rtype: str
        :return: String representing the estimated time to finish
        """
        self.current += increase
        return self._eta()


__all__ = ['find_numbers_in_text', 'constant_number', 'infix_to_postfix', 'filter_dict_by_keys',
           'FRACTIONAL_PATTERN', 'NUMBER_PATTERN', 'NUMBER_AND_FRACTION_PATTERN', 'ExpectedTimeToFinishCalculator']
