import re
from typing import Dict, Any, List, Tuple
from sympy.parsing.sympy_parser import stringify_expr, eval_expr, standard_transformations, \
    convert_equals_signs, convert_xor, split_symbols, implicit_multiplication, implicit_application

from .preproc import Preprocessor
from page.util import constant_number, infix_to_postfix
from page.const import PREP_KEY_ANS, PREP_KEY_EQN


DEFAULT_TRANSFORMATION = standard_transformations + \
                         (convert_equals_signs, convert_xor, split_symbols, implicit_multiplication,
                          implicit_application)


class ALG514Preprocessor(Preprocessor):
    def read_text(self, item: Dict[str, Any]) -> str:
        return item['sQuestion'].strip()

    def refine_answer(self, item: Dict[str, Any]) -> List[Tuple[str, ...]]:
        # Because the result in lSolutions represent a single pair of solution, we need to wrap it with tuple.
        return [tuple(x for x in item['lSolutions'])]

    def refine_formula_as_prefix(self, item: Dict[str, Any], numbers: List[dict]) -> List[Tuple[int, str]]:
        if item['Template'] is None:
            return []

        formula = [re.sub('([-+*/=])', ' \\1 ', eqn.lower().replace('-1', '1NEG')).replace('1NEG', '-1')
                   for eqn in item['Template']]  # Shorthand for linear formula
        tokens = re.split('\\s+', self.read_text(item))
        number_by_tokenid = {j: i for i, x in enumerate(numbers) for j in x['token']}

        # Build map between (sentence, token in sentence) --> number token index
        number_token_sentence = {}
        sent_id = 0
        sent_token_id = 0
        for tokid, token in enumerate(tokens):
            if token in '.!?':  # End of sentence
                sent_id += 1
                sent_token_id = 0
                continue

            if tokid in number_by_tokenid:
                number_token_sentence[(sent_id, sent_token_id)] = number_by_tokenid[tokid]

            sent_token_id += 1

        # [1] Build mapping between coefficients in the template and var names (N_0, T_0, ...)
        mappings = {}
        for align in item['Alignment']:
            var = align['coeff']
            val = align['Value']
            sent_id = align['SentenceId']
            token_id = align['TokenId']

            if (sent_id, token_id) not in number_token_sentence:
                # If this is not in numbers recognized by our system, regard it as a constant.
                positive, const_code = constant_number(val)
                mappings[var] = [const_code]
                if not positive:
                    mappings[var].append('-')

                continue

            number_id = number_token_sentence[(sent_id, token_id)]
            number_info = numbers[number_id]

            expression = ['N_%s' % number_id]
            expr_value = eval(number_info['value'])

            offset = 1
            while abs(val - expr_value) > 1E-10 and (sent_id, token_id + offset) in number_token_sentence:
                next_number_id = number_token_sentence[(sent_id, token_id + offset)]
                next_info = numbers[next_number_id]
                next_value = eval(next_info['value'])
                next_token = 'N_%s' % next_number_id

                if next_value >= 100:
                    # Multiplicative case: e.g. '[Num] million'
                    expr_value *= next_value
                    # As a postfix expression
                    expression.append(next_token)
                    expression.append('*')
                else:
                    # Additive case: e.g. '[NUM] hundred thirty-two'
                    expr_value += next_value
                    expression.append(next_token)
                    expression.append('+')

                offset += 1

            # Final check.
            # assert abs(val - expr_value) < 1E-5, "%s vs %s: \n%s\n%s" % (align, expr_value, numbers, item)
            mappings[var] = expression

        # [2] Parse template and convert coefficients into our variable names.
        # Free symbols in the template denotes variables representing the answer.
        new_formula = []
        free_symbols = []

        for eqn in formula:
            output_tokens = infix_to_postfix(eqn, mappings, free_symbols)

            if output_tokens:
                new_formula.append((PREP_KEY_EQN, output_tokens))

        if free_symbols:
            new_formula.append((PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))

        return new_formula


__all__ = ['ALG514Preprocessor']
