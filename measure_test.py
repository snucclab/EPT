import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from page.sympy.transform import AnswerChecker
from page.torch.text_field import ProblemTextField
from page.torch.eq_field import *
from page.const import FORMAT_NUM, PREP_KEY_ANS, NUM_PREFIX


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='set', type=str, nargs='+')
    parser.add_argument('--log', type=str, default='dataset')

    return parser.parse_args()


def read_dataset(path):
    with Path(path).open('r+t', encoding='UTF-8') as fp:
        for line in fp.readlines():
            if line.strip():
                item = json.loads(line)
                yield item['id'], item['text'], item['expr'], item['answer']


if __name__ == '__main__':
    args = parse_argument()

    logging.basicConfig(filename=str(Path(args.log, 'measure_test.log')),
                        level=logging.INFO, datefmt='%m/%d/%Y %H:%M:%S',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARN)

    # Prepare fields
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    text_field = ProblemTextField(tokenizer)
    field_to_test = [
        None,
        OpEquationField(['X_'], ['N_'], 'C_'),
        ExpressionEquationField(['X_'], ['N_'], 'C_', max_arity=2, force_generation=False),
        ExpressionEquationField(['X_'], ['N_'], 'C_', max_arity=2, force_generation=True)
    ]

    # Prepare answer checker
    seq_checker = AnswerChecker(is_expression_type=False)
    mem_checker = AnswerChecker(is_expression_type=True)

    for setname in tqdm(args.set):
        logging.info('\n========================================================\n'
                     'TEST for dataset: %s\n'
                     '\n'
                     '** Note: NoneType means original prefix equation in the dataset\n', setname)
        dataset = list(read_dataset(setname))

        for field in field_to_test:
            if field is not None:
                field.build_vocab([tpl[2] for tpl in dataset])

        error_counts = {field: 0 for field in field_to_test}

        for qid, text, expr, ans in tqdm(dataset):
            text = text_field.preprocess(text)
            numbers = text.number_value

            for field in field_to_test:
                if field is None:
                    e = [[FORMAT_NUM % int(tok[2:]) if tok.startswith(NUM_PREFIX) else tok
                          for typ, eqn in expr if typ != PREP_KEY_ANS for tok in eqn.split(' ')]]
                else:
                    e = field.process([field.preprocess(expr)])
                    e = field.convert_ids_to_equations(e)

                checker = mem_checker if isinstance(field, ExpressionEquationField) else seq_checker
                correct, result, error, system = checker.check(e[0], numbers, ans)
                checker = None

                if not correct:
                    logging.info("""
            ERROR OCCURRED @ %s
                    Field Type : %s
                    Original   : %s
                    Transformed: %s
                    Parsed     : %s
                    Expected   : %s
                    Resulted   : %s
                    Error? %s
                    Numbers? %s
                    """ % (qid, field.__class__.__name__, expr, e[0], system, ans, result, error,
                           {FORMAT_NUM % k: v['value'] for k, v in enumerate(numbers)}))

                    error_counts[field] += 1

        logging.info('\n----- SUMMARY for dataset: %s\n', setname)
        for field in field_to_test:
            logging.info('Total error in %25s: %5d' % (field.__class__.__name__, error_counts[field]))

    seq_checker.close()
    mem_checker.close()
