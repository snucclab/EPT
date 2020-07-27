import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from page.util import find_numbers_in_text


def write_problems(save_path: str, problems: List[dict]):
    path = Path(save_path + '.orig.jsonl')
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    # Write original dictionary with parsed field
    orig_wp = Path(save_path + '.orig.jsonl').open('w+', encoding='UTF-8')
    proc_wp = Path(save_path + '.jsonl').open('w+', encoding='UTF-8')

    for prob in problems:
        orig_wp.write(json.dumps(prob) + '\n')

        del prob['original']
        proc_wp.write(json.dumps(prob) + '\n')

    orig_wp.close()
    proc_wp.close()


class Preprocessor:
    def __init__(self):
        self.conversion_list = []

    def add_conversion(self, source: str, destination: str, set_name: str, key_field: str = None,
                       include_keys: List[int] = None, exclude_keys: List[int] = None):
        assert include_keys is None or exclude_keys is None, \
            "Include/Exclude keys cannot be 'not None' at the same time!"
        self.conversion_list.append((str(source), str(destination), set_name, key_field, include_keys, exclude_keys))

    def add_fold_information(self, source: str, destination_prefix: str, set_name: str, key_field: str,
                             folds: List[List[int]]):
        for i, fold in enumerate(folds):
            fold_name = '_fold%s' % i
            self.add_conversion(source, str(destination_prefix) + fold_name + '_train',
                                set_name + fold_name + '_train', key_field,
                                exclude_keys=fold)
            self.add_conversion(source, str(destination_prefix) + fold_name + '_test',
                                set_name + fold_name + '_test', key_field,
                                include_keys=fold)

    def read_text(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def refine_answer(self, item: Dict[str, Any]) -> List[Tuple[str, ...]]:
        raise NotImplementedError()

    def refine_formula_as_prefix(self, item: Dict[str, Any], numbers: List[dict]) -> List[Tuple[int, str]]:
        raise NotImplementedError()

    def refine_problem_text(self, item: Dict[str, Any]) -> Tuple[str, List[dict]]:
        # Read problem text & Normalize it.
        text = self.read_text(item)
        return find_numbers_in_text(text)

    def convert(self, item: Dict[str, Any]) -> Dict[str, Any]:
        new_item = {'original': item}

        text, numbers = self.refine_problem_text(item)
        new_item['text'] = text
        new_item['numbers'] = numbers

        answer_list = self.refine_answer(item)
        new_item['answer'] = answer_list

        prefix_formula = self.refine_formula_as_prefix(item, numbers)
        new_item['expr'] = prefix_formula

        return new_item

    def visit_problems(self, set_name: str, problems: List[dict], index_field: Optional[str] = None,
                       include_indices: List[str] = None, exclude_indices: List[str] = None) -> List[dict]:
        from tqdm import tqdm  # To show the progress

        parsed_problems = []
        for prob_index, prob in tqdm(enumerate(problems), desc='Problems', total=len(problems), mininterval=2):
            written_index = prob[index_field] if index_field is not None else prob_index

            if include_indices is not None and written_index not in include_indices:
                continue
            if exclude_indices is not None and written_index in exclude_indices:
                continue

            prob = self.convert(prob)
            prob['id'] = '%s-%05d' % (set_name, written_index)
            prob['set'] = set_name  # Assign set name in parsed.

            parsed_problems.append(prob)

        return parsed_problems

    def run(self):
        for src, dst, name, key, include, exclude in self.conversion_list:
            # For each set,
            with Path(src).open(encoding='UTF-8') as fp:
                problems = json.load(fp)

            problems = self.visit_problems(name, problems, key, include, exclude)
            write_problems(dst, problems)


__all__ = ['Preprocessor']
