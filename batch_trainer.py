import argparse
import logging
from itertools import groupby
from json import dumps as json_dump
from os import cpu_count, environ
from pathlib import Path
from time import time, sleep

from numpy import mean
from ray import remote, init, shutdown
from ray.util import ActorPool
from scipy.stats import sem

from page.config import TrainerConfig
from page.torch.trainer import Trainer
from page.torch.util import get_available_device_count
from page.util import ExpectedTimeToFinishCalculator


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--set', type=str)
    parser.add_argument('--config', type=str, nargs='+')
    parser.add_argument('--ray-args', '-r', type=str, default='')
    parser.add_argument('--model', type=str, choices=['vanilla', 'expr', 'ept'], nargs='+',
                        default=['vanilla', 'expr', 'ept'])

    return parser.parse_args()


@remote
class RayBatchActor(object):
    def train(self, kwargs):
        trainer = Trainer(kwargs['name'] + '/' + kwargs['runname'],
                          TrainerConfig.from_pretrained(kwargs['config_path']).copy(**kwargs['config_copy']),
                          kwargs['data_path'][0], kwargs['data_path'][-1], dev=kwargs['data_path'][1],
                          disable_dataparallel=not kwargs['data_parallel'])

        time_begin = time()
        trainer.train()
        time_delta = time() - time_begin

        results = [int(x) for x in trainer.get_evaluation_output('Dev')]
        dev_result = sum(results) / len(results) * 100

        devmax_result = trainer.get_metrics()['Dev/correct_max'] * 100

        results = [int(x) for x in trainer.get_evaluation_output('Test')]
        test_result = sum(results) / len(results) * 100

        trainer.close()
        return [kwargs['name'], kwargs['runname'], time_delta, dev_result, devmax_result, test_result]


if __name__ == '__main__':
    args = parse_argument()
    gpus = get_available_device_count(default=0)
    cpus = cpu_count()

    """ Build Dataset paths """
    set_path = args.set
    setname = Path(set_path).name
    if not Path('runs').exists():
        Path('runs').mkdir(parents=True)

    datapairs = []
    if Path(set_path + '_train.jsonl').exists():
        train = set_path + '_train.jsonl'
        test = set_path + '_test.jsonl'
        if Path(set_path + '_dev.jsonl').exists():
            dev = set_path + '_dev.jsonl'
        else:
            dev = None

        datapairs.append((train, dev, test, 1, 'seed1'))
    elif Path(set_path + '_fold0_train.jsonl').exists():
        seed = 1
        fold = 0
        while True:
            if not Path(set_path + '_fold%s_train.jsonl' % fold).exists():
                break
            train = set_path + '_fold%s_train.jsonl' % fold
            test = set_path + '_fold%s_test.jsonl' % fold

            # Cross-validation does not need seed change
            datapairs.append((train, None, test, 1, 'fold%s' % fold))
            fold += 1
    else:
        print('No dataset found!')
        exit(1)

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path('runs', setname + '.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                                datefmt='%m/%d/%Y %H:%M:%S'))

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.root.setLevel(logging.FATAL)
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARN)

    logger = logging.getLogger('BatchTrainer')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Initialize ray
    ray_args = environ.get('OPT_FOR_RAY', args.ray_args)
    if not ray_args:
        init()
    else:
        init(**eval(ray_args))

    eta = ExpectedTimeToFinishCalculator(len(args.config))
    experiment_map = {}

    for config_path in args.config:
        base_config = TrainerConfig.from_pretrained(config_path)
        base_name = Path(config_path).name.split('.')[0]

        # Setup experiments
        experiments = []
        for model_type in args.model:
            config = base_config.copy(model_type=model_type)
            model = config.model
            name = 'runs/%s-%s/%s' % (setname, config.epoch, model.experiment_name)
            logger.info('Experiment %s will use the following configuration:\n%s',
                        name, json_dump(config.to_kwargs()))

            for train, dev, test, seed, runname in datapairs:
                experiments.append({
                    'name': name, 'runname': runname + '-' + base_name, 'config_path': config_path,
                    'config_copy': {'seed': seed, 'model_type': model_type},
                    'data_path': [train, dev, test],
                    'data_parallel': True  # 'xlarge' not in model.encoder_model
                })

        # Setup ray pool
        if gpus:
            if '-base' in base_config.model.encoder_model:
                num_gpu = 0.5
            elif '-large' in base_config.model.encoder_model:
                num_gpu = 1.0
            else:
                num_gpu = 2.0
        else:
            num_gpu = 0.0

        if num_gpu not in experiment_map:
            experiment_map[num_gpu] = []

        experiment_map[num_gpu] += experiments

    for num_gpu, experiments in experiment_map.items():
        max_available_actors = min(cpus // 2, gpus // num_gpu if gpus else cpus)
        pool = ActorPool([RayBatchActor.options(num_cpus=2, num_gpus=num_gpu).remote()
                          for _ in range(int(max_available_actors))])

        exp_result = pool.map_unordered(lambda actor, kwargs: actor.train.remote(kwargs), experiments)
        for name, results in groupby(sorted(exp_result, key=lambda t: t[0]), key=lambda t: t[0]):
            results = list(results)

            # Log the results
            logger.info('Experiment: %s ------------------', name)
            for name, runname, timedelta, devresult, devmaxresult, testresult in results:
                logger.info('\t%s training time: %10.3f', runname, timedelta)
                logger.info('\t%s dev. accuracy: %7.3f', runname, devresult)
                logger.info('\t%s dev. max. acc: %7.3f', runname, devmaxresult)
                logger.info('\t%s test accuracy: %7.3f', runname, testresult)

            # Write the average result
            if len(results) > 1:
                _, _, time_delta, dev_results, devmax_results, test_results = zip(*results)
                logger.info('\ttime average %7.3f±%7.3f', mean(time_delta), sem(time_delta))
                logger.info('\tDEV. average %7.3f±%7.3f', mean(dev_results), sem(dev_results))
                logger.info('\tDEV. max avg %7.3f±%7.3f', mean(devmax_results), sem(devmax_results))
                logger.info('\tTEST average %7.3f±%7.3f', mean(test_results), sem(test_results))
            logger.info('--------------------------------------------------')

        # Cool down GPUs
        del pool
        sleep(180)
        logger.info('\tExpected time to finish BATCH TRAINER: %s', eta.step())

    # Shutdown ray.
    shutdown()
