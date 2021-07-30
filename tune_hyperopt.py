import argparse
import logging
from json import dumps as json_dump, dump as json_file_dump
from os import cpu_count, environ
from pathlib import Path
from shutil import rmtree
from time import sleep

from ray import tune, init, shutdown
from ray.tune import Trainable
from ray.tune.result import DONE
from ray.tune.schedulers import AsyncHyperBandScheduler, PopulationBasedTraining, MedianStoppingRule

from page.config import TrainerConfig
from page.torch.trainer import Trainer
from page.torch.util import get_available_device_count
from page.util import ExpectedTimeToFinishCalculator


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-T', type=int)
    parser.add_argument('--sample', '-n', type=int, default=1)
    parser.add_argument('--set', '-s', type=str, nargs='+')
    parser.add_argument('--config', '-c', type=str, nargs='+')
    parser.add_argument('--opt-space', '-o', type=str, nargs='+')
    parser.add_argument('--ray-args', '-r', type=str, default='')
    parser.add_argument('--trial-scheduler', '-t', type=str, choices=['hyperband', 'pbt', 'median', 'none'],
                        default='none')

    return parser.parse_args()


def get_set_path(set_path):
    set_path = str(Path(set_path).absolute())

    return {
        'train': set_path + '_train.jsonl',
        'dev': set_path + '_dev.jsonl' if Path(set_path + '_dev.jsonl').exists() else set_path + '_test.jsonl'
    }


def parse_option_space(args):
    sampling_space = {'trainer_path': 'runs'}
    perturb_space = {}

    for opt in args.opt_space:
        key, optstr = opt.split('=')
        space_opt = eval(optstr)

        if isinstance(space_opt, (int, float)):
            sampling_space[key] = space_opt
        elif type(space_opt) is list:
            sampling_space[key] = tune.grid_search(space_opt)
            perturb_space[key] = space_opt
        elif type(space_opt) is str:
            sampling_space[key] = tune.sample_from(eval('lambda spec: ' + space_opt))
            perturb_space[key] = eval('lambda: ' + space_opt)
        else:
            sampling_space[key] = tune.sample_from(space_opt)
            perturb_space[key] = space_opt

    return sampling_space, perturb_space


class RayTrainer(Trainable):
    def _setup(self, config):
        super()._setup(config)

        config_path = config.pop('config_path')
        set_path = config.pop('set_path')
        trainer_path = config.pop('trainer_path')
        data_parallel = config.pop('data_parallel')

        self._trainer = Trainer(trainer_path, TrainerConfig.from_pretrained(config_path).copy(**config),
                                set_path['train'], set_path['dev'], dev=set_path['dev'],
                                disable_dataparallel=not data_parallel)
        self._trainer.set_seed()

    def _train(self):
        self._trainer.run_a_chkpt_iter()
        metrics = {key[4:]: value
                   for key, value in self._trainer.get_metrics().items() if key.startswith('Dev/')}
        metrics['progress'] = self._trainer.current_epoch

        if self._trainer.is_done:
            metrics[DONE] = True

        return metrics

    def _save(self, tmp_checkpoint_dir):
        # Trainer already have functionality to checkpoint the model.
        self._trainer.checkpoint()
        return str(Path(tmp_checkpoint_dir))

    def _restore(self, checkpoint):
        self._trainer.restore_checkpoint()

    def _stop(self):
        self._trainer.rotate_checkpoint(1)
        self._trainer.close()
        super()._stop()


if __name__ == '__main__':
    args = parse_argument()
    sampling_space, perturb_space = parse_option_space(args)

    run_dir = Path('runs')
    if not run_dir.exists():
        run_dir.mkdir(parents=True)

    # Compute the number of devices per trial
    gpus = get_available_device_count(default=0)
    cpus = int(cpu_count() * 0.75)
    resource_per_trial = {'cpu': 2}

    # Initialize ray
    ray_args = environ.get('OPT_FOR_RAY', args.ray_args)
    if not ray_args:
        init()
    else:
        init(**eval(ray_args))

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path('runs', 'tune.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)

    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S', level=logging.INFO)
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARN)

    logger = logging.getLogger('HyperOpt')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('Executing hyperparameter optimization with space: %s', args.opt_space)

    eta = ExpectedTimeToFinishCalculator(len(args.config) * len(args.set))

    # Execute random search
    for conf_path in args.config:
        conf_path = Path(conf_path).absolute()
        base_path = str(conf_path) + '.tmp'

        base_conf = TrainerConfig.from_pretrained(conf_path) \
            .copy(epoch=args.epoch, epoch_report=args.epoch // 50, epoch_chkpt=args.epoch // 20)
        base_conf.save_pretrained(base_path, enforce_path=True)

        sampling_space['config_path'] = base_path
        sampling_space['data_parallel'] = True  # False

        if gpus:
            if '-base' in base_conf.model.encoder_model:
                resource_per_trial['gpu'] = 0.5
            elif '-large' in base_conf.model.encoder_model:
                resource_per_trial['gpu'] = 1.0
            else:
                resource_per_trial['gpu'] = 2.0
                # sampling_space['data_parallel'] = True

        for set_path in args.set:
            setname = Path(set_path).name.split('_')[0]
            sampling_space['set_path'] = get_set_path(set_path)
            logger.info('Search hyper-parameters for %s based on configuration at %s', set_path, conf_path)

            exp_name = '%s-%s' % (setname, base_conf.model.experiment_name)
            if args.trial_scheduler == 'hyperband':
                scheduler = AsyncHyperBandScheduler(time_attr='progress', metric='correct', mode='max',
                                                    max_t=args.epoch, grace_period=args.epoch / 10,
                                                    reduction_factor=2, brackets=4)
            elif args.trial_scheduler == 'median':
                scheduler = MedianStoppingRule(time_attr='progress', metric='correct',
                                               grace_period=args.epoch / 10, min_time_slice=args.epoch / 2)
            elif args.trial_scheduler == 'pbt':
                scheduler = PopulationBasedTraining(time_attr='progress', metric='correct', mode='max',
                                                    perturbation_interval=args.epoch / 4,
                                                    hyperparam_mutations=perturb_space)
            else:
                scheduler = None

            logger.info('Trial scheduler used: %s', str(scheduler))
            analysis = tune.run(RayTrainer, name=exp_name,
                                config=sampling_space, local_dir='runs', num_samples=args.sample, scheduler=scheduler,
                                resources_per_trial=resource_per_trial, raise_on_failed_trial=False)

            # Remove base path
            Path(base_path).unlink()

            logger.info('Trial informations')
            dataframe = analysis.dataframe().sort_values(by='correct', ascending=False)
            for _, record in dataframe.iterrows():
                tag = record['experiment_tag'].split('_', 1)[1]
                logger.info('\tTrial %3d (%-40s): Correct %.4f (EMA %.4f, MAX %.4f) / Stop at %4.0f (%9.4fs)',
                            record['trial_id'], tag, record['correct'],
                            record['correct_ema'], record['correct_max'],
                            record['progress'], record['time_total_s'])

            best_config = base_conf.copy(**analysis.get_best_config('correct', scope='last')).to_kwargs()
            logger.info('BEST SETUP FOR %s: %s', set_path, json_dump(best_config))

            conf_path = Path(conf_path)
            save_path = Path(conf_path.parent, setname, conf_path.name)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)

            logger.info('\tThis will be saved @ %s', save_path)
            with save_path.open('w+t', encoding='UTF-8') as fp:
                json_file_dump(best_config, fp)

            logger.info('\tExpected time to finish HYPER-PARAMETER SEARCH: %s', eta.step())
            logger.info('========================================================================')

            # Remove all recordings to reduce disk space
            for path in Path('runs', exp_name).glob('**/%s' % sampling_space['trainer_path']):
                for d in path.glob('*'):
                    if d.is_dir():
                        rmtree(d)

            sleep(180)  # Sleep for 3 minutes to cool down GPU/CPUs

    # Shutdown ray
    shutdown()
