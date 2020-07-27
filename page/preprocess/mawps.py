import re
from typing import Dict, Any, List, Tuple

from page.const import PREP_KEY_ANS, PREP_KEY_EQN
from page.util import infix_to_postfix
from .preproc import Preprocessor


class MAWPSPreprocessor(Preprocessor):
    def read_text(self, item: Dict[str, Any]) -> str:
        masked_text = re.sub('\\s+', ' ', item['mask_text']).strip().split(' ')
        temp_tokens = item['num_list']

        regenerated_text = []
        for token in masked_text:
            if token.startswith('temp_'):
                regenerated_text.append(str(temp_tokens[int(token[5:])]))
            else:
                regenerated_text.append(token)

        return ' '.join(regenerated_text)

    def refine_answer(self, item: Dict[str, Any]) -> List[Tuple[str, ...]]:
        return [(x,) for x in item['lSolutions']]

    def refine_formula_as_prefix(self, item: Dict[str, Any], numbers: List[dict]) -> List[Tuple[int, str]]:
        template_to_number = {}
        template_to_value = {}
        number_by_tokenid = {j: i for i, x in enumerate(numbers) for j in x['token']}

        for tokid, token in enumerate(re.sub('\\s+', ' ', item['mask_text']).strip().split(' ')):
            if token.startswith('temp_'):
                assert tokid in number_by_tokenid, (tokid, number_by_tokenid, item)

                num_id = number_by_tokenid[tokid]
                template_to_number[token] = ['N_%s' % num_id]
                template_to_value[token] = numbers[num_id]['value']

        # We should read both template_equ and new_equation because of NONE in norm_post_equ.
        formula = item['template_equ'].split(' ')
        original = item['new_equation'].split(' ')
        assert len(formula) == len(original)

        # Recover 'NONE' constant in the template_equ.
        for i in range(len(formula)):
            f_i = formula[i]
            o_i = original[i]

            if f_i == 'NONE':
                formula[i] = original[i]
            elif f_i.startswith('temp_'):
                assert abs(float(template_to_value[f_i]) - float(o_i)) < 1E-4,\
                    "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)
            else:
                # Check whether two things are the same.
                assert f_i == o_i, "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)

        free_symbols = []
        new_formula = [(PREP_KEY_EQN, infix_to_postfix(formula, template_to_number, free_symbols))]

        if free_symbols:
            new_formula.append((PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))

        return new_formula


__all__ = ['MAWPSPreprocessor']
