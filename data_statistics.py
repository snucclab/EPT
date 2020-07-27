import logging
from pathlib import Path

from page.config import TrainerConfig

if __name__ == '__main__':
    # Enable logging system
    file_handler = logging.FileHandler(filename=Path('dataset', 'stat.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                                datefmt='%m/%d/%Y %H:%M:%S'))

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.root.setLevel(logging.FATAL)
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARN)

    logger = logging.getLogger('DataStat')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    for setname in ['alg514', 'draw', 'mawps']:
        config = TrainerConfig.from_pretrained(Path('config', setname, 'base.json'))
        logger.info('Dataset: %s', setname)

        if setname != 'draw':
            setname += '_fold0'

        setpath = 'dataset/' + setname
        datasets = [setpath + '_train.jsonl', setpath + '_test.jsonl']
        if Path(setpath + '_dev.jsonl').exists():
            datasets += [setpath + '_dev.jsonl']
            datasets = config.read_datasets(*datasets)
        else:
            train, _, test = config.read_datasets(*datasets)
            datasets = [train, test]

        stats = {}
        for data in datasets:
            stat = data.get_item_statistics()

            for key in stat:
                if key not in stats:
                    stats[key] = []

                stats[key] += stat[key]

        lengths = stats['text_token']
        logger.info('Information about lengths of text sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / len(lengths))

        lengths = stats['text_number']
        logger.info('Information about the number of numbers in text: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / len(lengths))

        lengths = stats['eqn_op_token']
        logger.info('Information about lengths of token unit sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / len(lengths))

        lengths = stats['eqn_expr_token']
        logger.info('Information about lengths of operator unit sequences: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / len(lengths))

        lengths = stats['eqn_unk']
        logger.info('Information about the number of unknowns used: Range %s - %s (mean: %s)',
                    min(lengths), max(lengths), sum(lengths) / len(lengths))

        logger.info('===========================================================================')
