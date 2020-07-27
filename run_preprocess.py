import argparse
import json
import logging
from pathlib import Path


def preprocess_each():
    global args
    from page.preprocess.alg514 import ALG514Preprocessor
    from page.preprocess.mawps import MAWPSPreprocessor

    """
    *********************************
    **** Preprocess each dataset ****
    *********************************
    """
    # Preprocess ALG514
    if not Path(args.output, 'alg514_fold0_test.jsonl').exists():
        print("\033[1;44;38mBegin to process ALG514.\033[0m")
        preproc = ALG514Preprocessor()

        # Add ALG514 path
        path = Path(args.input, args.alg)
        folds = []
        for i in range(5):
            with Path(path.parent, 'fold-%s.txt' % i).open(encoding='UTF-8') as fp:
                indices = [int(line.strip()) for line in fp.readlines() if line.strip()]
            folds.append(indices)

        preproc.add_fold_information(path, Path(args.output, 'alg514'), 'alg514', key_field='iIndex', folds=folds)
        preproc.run()
    else:
        print("\033[1;32mALG514 already exists\033[0m")

    # Preprocess DRAW
    if not Path(args.output, 'draw_test.jsonl').exists():
        print("\033[1;44;38mBegin to process DRAW.\033[0m")
        preproc = ALG514Preprocessor()

        # Add Draw path
        path = Path(args.input, args.draw)
        for key in ['train', 'test', 'dev']:
            with Path(path.parent, 'draw-%s.txt' % key).open(encoding='UTF-8') as fp:
                indices = [int(line.strip()) for line in fp.readlines() if line.strip()]
            preproc.add_conversion(path, Path(args.output, 'draw_%s' % key), 'draw_%s' % key,
                                   key_field='iIndex', include_keys=indices)

        preproc.run()
    else:
        print("\033[1;32mDRAW already exists\033[0m")

    # Preprocess MAWPS
    if not Path(args.output, 'mawps_fold4_test.jsonl').exists():
        print("\033[1;44;38mBegin to process MAWPS.\033[0m")
        preproc = MAWPSPreprocessor()

        # Add MAWPS path
        path = Path(args.input, args.mawps)
        folds = []
        with path.open(encoding='UTF-8') as rp:
            fold_problems = json.load(rp)['test_5_fold']
            all_problems = []

            for problems in fold_problems:
                all_problems += problems
                folds.append([item['iIndex'] for item in problems])

        path = Path(path.parent, 'flat.' + path.name)
        if not path.exists():
            with path.open('w+t', encoding='UTF-8') as fp:
                json.dump(all_problems, fp)

        preproc.add_fold_information(path, Path(args.output, 'mawps'), 'mawps', key_field='iIndex', folds=folds)
        preproc.run()
    else:
        print("\033[1;32mMAWPS already exists\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--input', '-in', '-I', type=str, required=True, help='Path of input files')
    parser.add_argument('--output', '-out', '-O', type=str, required=True, help='Path to save output files')

    parser.add_argument('--alg', type=str, help='Path of ALG514 files in input directory',
                        default='alg514/alg514.json')
    parser.add_argument('--draw', type=str, help='Path of ALG514 files in input directory',
                        default='draw/draw.json')
    parser.add_argument('--mawps', type=str, help='Path of MAWPS files in input directory',
                        default='mawps/mawps.json')

    # Parse arguments
    args = parser.parse_args()

    outpath = Path(args.output)
    if not outpath.exists():
        outpath.mkdir(parents=True)

    logging.basicConfig(filename=str(Path(args.output, 'preprocess.log')),
                        level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s @ %(name)s:%(funcName)s :: %(message)s')
    logger = logging.getLogger('Preprocess')

    preprocess_each()
