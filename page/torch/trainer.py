import logging
import math
import random
import shutil
from os import environ as ENV
from pathlib import Path
from typing import List

import numpy
import torch
from torch.nn import DataParallel
from transformers import get_linear_schedule_with_warmup

from page.config import *
from page.const import *
from page.model.model import Solver
from page.sympy.transform import AnswerChecker
from page.util import ExpectedTimeToFinishCalculator
from .dataset import ProblemInstance
from .util import required_space_param, get_available_device_count


def _accumulate_stats(values: dict, accumulator: dict):
    """
    Accumulate statistics

    :param dict values: Dictionary of values to accumulate
    :param dict accumulator: Target dictionary for accumulation
    """
    for k, v in values.items():
        if k not in accumulator:
            accumulator[k] = []

        # If this is a primitive values, just append it. Otherwise, we need to convert to primitive ones.
        if isinstance(v, (float, int, bool, str)):
            accumulator[k].append(v)
        else:
            accumulator[k] += v.tolist() if v.numel() > 1 else [v.item()]


def _write_csv_line(fp, *values):
    """
    Write a CSV line

    :param fp: File handle where values will be written
    :param values: Variable length arguments to write
    """
    strings = []
    for value in values:
        if type(value) is str:
            # Wrap string with quotes
            value = "\"%s\"" % value.replace('"', '\\"')
        elif type(value) is float:
            # Write float as a fixed-point decimal number
            value = "%.6f" % value
        else:
            value = str(value)
        strings.append(value)

    fp.write(','.join(strings) + '\n')


def _unwrap_parallel(obj):
    """
    Unwrap DataParallel wrapper class if exists.

    :param obj: A torch.nn.Module object
    :return: Unwrapped module.
    """
    return obj.module if isinstance(obj, DataParallel) else obj


def _normalize_gradients(*parameters):
    """
    Normalize gradients (as in NVLAMB optimizer)

    :param parameters: List of parameters whose gradient will be normalized.
    :return: Frobenious Norm before applying normalization.
    """
    parameters = [p for p in parameters if p.grad is not None]

    # Compute total Frobenius norm
    total_norm = 0
    for p in parameters:
        total_norm += p.grad.data.norm(2.0).item() ** 2.0
    total_norm = total_norm ** 0.5

    # Compute normalization constant. Set 1E-12 for minimum value to avoid inf.
    normalizer = 1.0 / max(total_norm, 1e-12)
    for p in parameters:
        p.grad.data.mul_(normalizer)

    return total_norm


class Trainer(object):
    """
    Trainer class
    """

    def __init__(self, chkpt_path: str, config: TrainerConfig,
                 train: str, test: str, dev: str = None, disable_dataparallel: bool = False):
        """
        Instantiate trainer

        :param str chkpt_path: Path to checkpoint the model, optimizer and scheduler
        :param TrainerConfig config: Configuration instance for Trainer
        :param str train: Path to JSON with lines file which contains the training set
        :param str test: Path to JSON with lines file which contains the evaluation set
        :param str dev: Path to JSON with lines file which contains the development set (optional)
        :param bool disable_dataparallel:
            True if module should not be parallelized across different GPU devices. False by default.
        """
        # Register configuration
        self._config = config
        self.disable_dataparallel = disable_dataparallel

        # Prepare internal states
        self._best_on_dev = 0.0  #: Best score on the development set
        self._ema_on_dev = None  #: Exponential Moving Average score on the development set.
        self._random_restored = False  #: Whether the RNG state restored or not

        # Epoch & step information
        self._epoch = 0
        self._steps_to_go = 0
        self._step_per_epoch = 0
        self._minibatch_per_epoch = 0

        # Dictionary that records the last performance metrics
        self._last_performances = {}
        self._last_metrics = {}

        # Prepare checkpointing
        self._chkpt_path = Path(chkpt_path)
        if not self._chkpt_path.exists():
            self._chkpt_path.mkdir(parents=True)

        # Logging file handler
        file_handler = logging.FileHandler(filename=Path(chkpt_path, 'train.log'), encoding='UTF-8')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                                                    datefmt='%m/%d/%Y %H:%M:%S'))
        file_handler.setLevel(logging.INFO)

        # Set the logger
        self._logger = logging.getLogger(self.__class__.__name__ + '_%s' % id(self))
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)

        # If DEBUG is on, turn on the anomaly detection
        if 'DEBUG' in ENV:
            torch.autograd.set_detect_anomaly(True)

        # Prepare Tensorboard if available.
        try:
            from tensorboardX import SummaryWriter
            self._writer = SummaryWriter(logdir=str(self._chkpt_path), flush_secs=30)
        except ImportError:
            self._writer = None

        # Prepare data-parallel if available.
        if torch.cuda.is_available():
            devices = get_available_device_count()
            cuda_keys = list(range(devices))
            random.shuffle(cuda_keys)

            self.main_device = torch.device('cuda', cuda_keys[0])
            self.device_order = cuda_keys
        else:
            self.main_device = torch.device('cpu')
            self.device_order = [self.main_device]
        self._logger.info("We will use [%s] device as a main device for training, with ordering [%s]",
                          self.main_device, self.device_order)

        # Read the datasets
        self.set_seed()  #: Set seed before loading the datasets (because of shuffling in training set)
        self.trainset, self.devset, self.evalset = self._config.read_datasets(train=train, dev=dev, test=test)
        self._trainit = iter(self.trainset)

        # Log dataset statistics
        self._logger.info('From %s, we loaded %s mini-batch(es)', train, len(self.trainset))
        self._logger.info('From %s, we loaded %s mini-batch(es)', dev, len(self.devset))
        self._logger.info('From %s, we loaded %s mini-batch(es)', test, len(self.evalset))
        self.trainset.print_item_statistics(self._logger)

        # Build or restore module
        self._module = None
        self._module_init = {}
        self._optimizer = None
        self._answer_checker = None
        self.restore_checkpoint()

    @property
    def checkpoints(self) -> List[Path]:
        """
        :rtype: List[Path]
        :return: List of checkpointed steps (dictionaries)
        """
        checkpoints = sorted(Path(self._chkpt_path).glob('*'))
        checkpoints = [x for x in checkpoints if x.is_dir() and x.name.isnumeric()]
        return checkpoints

    @property
    def last_checkpoint(self) -> Path:
        """
        :rtype: Path
        :return: The last checkpoint if exists. Otherwise, None
        """
        return self.checkpoints[-1] if len(self.checkpoints) else None

    @property
    def current_epoch(self) -> int:
        """
        :rtype: int
        :return: Current epoch index
        """
        return self._epoch

    @property
    def is_done(self) -> bool:
        """
        :rtype: bool
        :return: True if trainer already reached maximum epoch specified.
        """
        return self._epoch == self._config.epoch

    def close(self):
        """
        Close and clean-up the trainer.
        """
        if self._writer is not None:
            # Close the TensorboardX
            self._writer.close()
            self._writer = None
        if self._answer_checker is not None:
            # Kill the answer checker child processes
            self._answer_checker.close()
            self._answer_checker = None

    def rotate_checkpoint(self, max_item: int = 10):
        """
        Rotate checkpoints

        :param int max_item: Maximum number of allowed checkpoints
        """
        # Check if we should delete older checkpoint(s)
        if len(self.checkpoints) <= max_item:
            return

        for chkpt in self.checkpoints[:-max_item]:
            # Remove old checkpoints
            self._logger.info("Deleting old checkpoint [%s]", chkpt)
            shutil.rmtree(chkpt)

    def checkpoint(self):
        """
        Make a checkpoint
        """
        # Build dictionary format to make the order directory names and the order of epoch index be the same.
        directory_format = '%%0%dd' % int(math.ceil(math.log10(self._config.epoch + 1)))
        # If directory exists, exit the method.
        output_dir = Path(self._chkpt_path, directory_format % self._epoch)
        if output_dir.exists():
            return

        # Prepare the directory for checkpointing
        self._logger.info("Save checkpoint to [%s]", output_dir)
        output_dir.mkdir(parents=True)

        # Save the all RNG states used in this trainer.
        torch.save({
            'numpy': numpy.random.get_state(),
            'random': random.getstate(),
            'trainset': self.trainset.get_rng_state(),
            'torch': {
                'cpu': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }, Path(output_dir, 'random.pt'))

        # Save Trainer's internal states
        torch.save({
            '_best_on_dev': self._best_on_dev,
            '_ema_on_dev': self._ema_on_dev,
            '_last_performances': self._last_performances,
            '_last_metrics': self._last_metrics
        }, Path(output_dir, 'internal.pt'))

        # Save the model
        _unwrap_parallel(self._module).save_pretrained(output_dir)
        # Save the optimizer
        torch.save(self._optimizer.state_dict(), Path(output_dir, 'optimizer.pt'))

        # Save the scheduler if available.
        if hasattr(self, '_scheduler'):
            torch.save(self._scheduler.state_dict(), Path(output_dir, 'scheduler.pt'))

        # Write configuration that has been used.
        self._config.save_pretrained(output_dir)
        # Rotate checkpoints.
        self.rotate_checkpoint()

    def restore_checkpoint(self):
        """
        Restore from the last checkpoint if available. Otherwise, configure this trainer from the scratch.
        """
        # Check if there exists any checkpoints.
        chkpt_path = self.last_checkpoint
        if chkpt_path:
            # reload configuration from the checkpoint
            self._config = TrainerConfig.from_pretrained(str(chkpt_path))
            self._logger.info("TrainerConfig at [%s] is restored.", chkpt_path)

            # Recover random number generator states
            self.set_seed()  # Set seed before restoring RNG
            random_path = Path(chkpt_path, 'random.pt')
            random_states = torch.load(random_path)
            numpy.random.set_state(random_states['numpy'])
            random.setstate(random_states['random'])
            self.trainset.set_rng_state(random_states['trainset'])

            torch.set_rng_state(random_states['torch']['cpu'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(random_states['torch']['cuda'])

            # Record that the RNG is restored.
            self._logger.info("State of random number generator is restored from [%s]", random_path)
            self._random_restored = True

            # Recover the trainer's internal states
            internal_states = torch.load(Path(chkpt_path, 'internal.pt'))
            for key, value in internal_states.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            self.set_seed()  # Set seed.

        # Build/restore model
        self._config.model.set_chkpt_path(chkpt_path)
        self._module = Solver.from_pretrained(config=self._config.model)
        self._module_init = {id(p): p.clone() for p in self._module.parameters()}
        self._module.to(self.main_device)
        self._logger.info("A network at [%s] is restored.", chkpt_path)

        # Compute the epoch/step information
        self._minibatch_per_epoch = len(self.trainset)
        self._step_per_epoch = int(math.ceil(self._minibatch_per_epoch / self._config.gradient_accumulation_steps))
        self._steps_to_go = self._step_per_epoch * self._config.epoch
        self._logger.info("Steps / Epoch = %5d", self._step_per_epoch)
        self._logger.info("We will run %3d epoch(s) or %6d step(s)", self._config.epoch, self._steps_to_go)
        self._logger.info("Per a single step, %2d gradient(s) will be accumulated. (Total %2d mini-batch(es)/epoch)",
                          self._config.gradient_accumulation_steps, self._minibatch_per_epoch)
        self._logger.info("We will report TRAINING loss/accuracy for every %3d epoch(s)", self._config.epoch_report)
        self._logger.info("We will report DEV ACC. and save CHKPTs for every %3d epoch(s)", self._config.epoch_chkpt)

        # Restore the number of steps that were passed before
        if chkpt_path:
            self._epoch = int(chkpt_path.name)
            self._logger.info("Attempt to restore from the checkpoint [%s]", chkpt_path)
            self._logger.info("Resume training from epoch %s", self._epoch)

        # Classify parameters to form parameter groups to build optimizer
        no_w_decay = {'bias', 'norm', 'Norm', '_embedding'}
        parameters = [((2 if 'text_model.model.embeddings' in n else (1 if 'text_model' in n else 0),
                        any(t in n for t in no_w_decay)), p)
                      for n, p in self._module.named_parameters()]
        parameters = groupby(sorted(parameters, key=lambda t: t[0]), key=lambda t: t[0])

        # Build optimizer groups
        optimizer_grouped_parameters = []
        for (encoder_type_flag, is_without_wd), group in parameters:
            group = {'params': [p for _, p in group]}

            if is_without_wd:
                group['weight_decay'] = 0.0

            if encoder_type_flag == 2 and self._config.fix_encoder_embedding:
                group['lr'] = 0.0
            elif encoder_type_flag == 1:
                group['lr'] = self._config.optimizer.kwargs['lr'] * self._config.lr_multiplier_encoder

            optimizer_grouped_parameters.append(group)

        # Build optimizer before restoration
        self._optimizer = self._config.optimizer.build(optimizer_grouped_parameters)
        self._logger.info("We will use the following optimizer: %s", self._optimizer)

        # Restore the optimizer if available.
        if chkpt_path:
            # Check if saved optimizer exists
            optimizer_file = Path(chkpt_path, 'optimizer.pt')
            if optimizer_file.is_file():
                self._optimizer.load_state_dict(torch.load(optimizer_file))
                self._logger.info("An optimizer for module at [%s] is restored.", optimizer_file)

        # Specify warmup strategy if warmup value is not negative
        warmup_steps = int(self._step_per_epoch * self._config.epoch_warmup)
        if warmup_steps >= 0:
            # Build scheduler before restoration
            self._scheduler = get_linear_schedule_with_warmup(self._optimizer, num_warmup_steps=warmup_steps,
                                                              num_training_steps=self._steps_to_go)
            self._logger.info("We will use linear scheduling: warm up %s epochs or %s steps",
                              self._config.epoch_warmup, warmup_steps)

            # Restore the scheduler if available
            if chkpt_path:
                # Check if saved scheduler exists
                scheduler_file = Path(chkpt_path, 'scheduler.pt')
                if scheduler_file.is_file():
                    self._scheduler.load_state_dict(torch.load(scheduler_file))
                    self._logger.info("A scheduler for module at [%s] is restored.", scheduler_file)

        # Log the threshold of gradient clipping.
        if self._config.gradient_clip > 0:
            self._logger.info("We will use gradient clipping at %.3f", self._config.gradient_clip)
        else:
            self._logger.info("We will not use gradient clipping")

        # Log the structure of the network.
        parameters_size = sum(p.numel() for p in self._module.parameters())
        disk_space = sum(required_space_param(p) for p in self._module.parameters())
        self._logger.info('==== [Network Structure] ====\n%s', str(self._module))
        self._logger.info('There are %12d parameters in a network. Required space for checkpointing is %.3fMB.',
                          parameters_size, disk_space / 1048576)

        # Wrap data parallel if we can use more than one GPU
        if len(self.device_order) > 1 and not self.disable_dataparallel:
            self._module = DataParallel(self._module, device_ids=self.device_order, output_device=self.device_order[0])
            self._logger.info("We identified [%s] devices for parallel training", len(self.device_order))
        else:
            self._logger.info("We don't use DataParallel.")

        # Set answer checker
        self._answer_checker = AnswerChecker(is_expression_type=_unwrap_parallel(self._module).is_expression_type,
                                             logger=self._logger)

    def set_seed(self):
        """
        Set the random seeds
        """
        if self._random_restored:
            # Ignore seed setting when state of rng was restored.
            return

        seed = self._config.seed
        self._logger.info("Seed for random number generation = %s", seed)

        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_evaluation_output(self, key: str):
        """
        Get the evaluation output of specified key.

        :param str key: metric key to read
        :return: metric value of specified key
        """
        return self._last_performances[key]

    def get_metrics(self) -> dict:
        """
        :return: The latest metric dictionary.
        """
        return self._last_metrics

    def run_a_chkpt_iter(self):
        """
        Run epochs until checkpointing
        """

        try:
            accumulated_values = {}

            for _ in range(self._config.epoch_chkpt):
                # For each epoch (at most the number of checkpointing epoch)
                self._epoch += 1

                all_grad_applied = True
                for batch_step in range(self._minibatch_per_epoch):
                    # For each minibatch
                    self._module.eval()
                    self._module.zero_grad()

                    # Load a minibatch
                    batch = next(self._trainit)
                    batch = self._load_batch(batch)

                    # Execute training
                    self._module.train()
                    reported_values = self._step(**batch)
                    reported_values['Loss/generate'] = reported_values['total_loss']
                    reported_values['total_loss'].backward()
                    all_grad_applied = False

                    # Accumulate statistics and update gradient
                    _accumulate_stats(reported_values, accumulated_values)
                    if (batch_step + 1) % self._config.gradient_accumulation_steps == 0:
                        self._update_grad()
                        all_grad_applied = True
                else:
                    # If there exists not-updated gradients, update gradient
                    if not all_grad_applied:
                        self._update_grad()

                if self._config.epoch_report > 0 and self._epoch % self._config.epoch_report == 0:
                    # Log metrics
                    if self._writer is not None:
                        for name, val in accumulated_values.items():
                            self._writer.add_scalar(name, sum(val) / len(val), self._epoch)
                        # Report current optimizer status
                        self._report_optimizer()

                    accumulated_values.clear()

            # Evaluate current result on development set
            self.evaluate()
            self.checkpoint()
        except Exception as e:
            self._logger.error('Exception occurred!', exc_info=e)
            raise e

    def train(self):
        """
        Do full-length training (until the maximum epoch)
        """
        # Set seed
        self.set_seed()

        # Prepare estimated time calculator class
        eta = ExpectedTimeToFinishCalculator(self._config.epoch, current=self._epoch)
        while self._epoch < self._config.epoch:
            self.run_a_chkpt_iter()
            eta_time = eta.step(increase=self._config.epoch_chkpt)
            self._logger.info('Expected time to finish: %s', eta_time)

        # Evaluate performance on the evaluation set
        try:
            self.evaluate(is_development=False)
        except Exception as e:
            self._logger.error('Exception occurred!', exc_info=e)
            raise e
        finally:
            # Remove old checkpoints and close Tensorboard writer
            self.rotate_checkpoint(1)

    def _update_grad(self):
        """
        Update accumulated gradients
        """
        if self._config.gradient_clip > 0:
            # If clipping threshold is set, then clip the gradient
            torch.nn.utils.clip_grad_norm_(self._module.parameters(), self._config.gradient_clip)

        if self._config.gradient_normalize:
            # If normalizing gradient is set, then normalize the gradient
            _normalize_gradients(*self._module.parameters())

        # Apply optimizer & scheduler
        self._optimizer.step()
        if hasattr(self, '_scheduler'):
            self._scheduler.step()

        # Reset the gradient
        self._module.zero_grad()

    def _load_batch(self, batch: ProblemInstance, is_training=True, max_len=0) -> dict:
        """
        Load batch instance into dictionary that can feed-able into the model.

        :param ProblemInstance batch: A mini-batch
        :param bool is_training: True if this batch is used for training. True by default.
        :param int max_len: Maximum length of equation to be generated. 0 by default (i.e. depends on the current batch)
        :rtype: dict
        :return: Dictionary representing mini-batch
        """
        # Prepare dictionary
        batch_dict = {'max_numbers': max(len(numbers) for numbers in batch.text.number_value),
                      IN_TXT: batch.text.token, IN_TPAD: batch.text.pad, IN_TNUM: batch.text.number}

        # Retrieve information about the target field
        required_field = _unwrap_parallel(self._module).required_field
        # Get equation in terms of the target field
        equation = getattr(batch, required_field)
        if is_training:
            # If this is training, then directly provide target equation for teacher-forcing
            batch_dict[IN_EQN] = equation
        else:
            # Otherwise, just provide information about maximum length of generation & arity of operators
            batch_dict['max_len'] = max(equation.shape[-2], max_len) + 1
            if required_field.startswith('tuple'):
                batch_dict['function_arities'] = getattr(self.evalset, required_field + '_field').function_arities

        if not isinstance(self._module, DataParallel):
            # If we applied data parallel, then move the value to the main device
            batch_dict = {k: v.to(self.main_device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_dict.items()}

        # Returned value is a dict.
        return batch_dict

    def _step(self, training: bool = True, **kwargs):
        """
        Execute forward computation of the module

        :param bool training: True if this execution is for training. True by default.
        :param kwargs: Keyword arguments to execute the module.
        :return: Result of execution.
            - If training is True, return value will be a dictionary mapping from string to accuracy/loss Tensors.
            - Otherwise, return value will be a LongTensor indicating the generated tokens
        """
        result = self._module(**kwargs)
        if type(result) is dict and training:
            return {k: v.mean() if training else v for k, v in result.items()}
        else:
            return result

    def _report_optimizer(self):
        """
        Report the current state of the optimizer
        """
        # Classify parameters by their types
        param_type = {id(p): ('Enc' if 'text_model.' in n else 'Dec') + ('Embed' if '_embedding' in n else 'Trans')
                      for n, p in _unwrap_parallel(self._module).named_parameters()}
        # Dictionary for accumulating parameter information
        param_states = {key: {'weight_norm': [], 'acc_update': []}
                        for key in set(param_type.values())}

        with torch.no_grad():
            # Without using gradients, accumulate information about weight and gradient
            for gid, group in enumerate(self._optimizer.param_groups):
                for p in group['params']:
                    id_p = id(p)
                    states = param_states[param_type[id_p]]
                    w_init = self._module_init[id_p]

                    w_elem = p.numel()
                    w_norm = p.norm(2).item() / w_elem
                    delta_norm = (w_init - p.clone().cpu()).norm(2).item() / w_elem

                    states['weight_norm'].append(w_norm)
                    states['acc_update'].append(delta_norm)

        # Write accumulated results
        if self._writer:
            for part, states in param_states.items():
                prefix = 'Optimizer_%s/%%s' % part

                for key, val in states.items():
                    if not len(val):
                        continue

                    # Track average & histograms
                    val = numpy.array(val)
                    self._writer.add_scalar(prefix % key, val.mean(), self._epoch)
                    self._writer.add_scalar(prefix % (key + '_std'), val.std(), self._epoch)

    def _check_equation(self, checker: AnswerChecker, outputs: torch.Tensor, batch: ProblemInstance):
        """
        Verify whether the outputted equation is correct or not.

        :param AnswerChecker checker: AnswerChecker instance to compute equation and check answer
        :param torch.Tensor outputs:
            LongTensor containing generated equations.
            - If the model should generate op-tokens, Shape = [B, M, T], where B = batch size, M = beams, and T = length
            - Otherwise, Shape = [B, M, T, 1+2A], where A = maximum arity.
        :param batch:
        :return:
        """
        # Retrieve size information
        batch_sz, beam_sz = outputs.shape[:2]

        # Get the target field information
        required_field = _unwrap_parallel(self._module).required_field
        # Retrieve the target field
        field = getattr(self.evalset, required_field + '_field')
        # Recover string representation of gold set and generated beams
        golds = field.convert_ids_to_equations(getattr(batch, required_field))
        beams = [field.convert_ids_to_equations(outputs[i]) for i in range(batch_sz)]

        outputs = []
        for i in range(batch_sz):
            # For each batch, retrieve information about written numbers and expected answer tuples
            numbers = batch.text.number_value[i]
            expected = batch.expected[i]

            # Test whether the produced equation in each beam
            results = [checker.check(beam, numbers, expected) for beam in beams[i]]
            # Record outputs: (index, goldset output, generated output, correctness)
            outputs.append((i, golds[i], beams[i], results))

        return outputs

    def evaluate(self, is_development: bool = True):
        """
        Evaluate the current model.

        :param bool is_development: True if current evaluation is done on development set. True by default.
        """
        # Shortcut for beam size
        beam_size = self._config.model.beam_size
        # Accumulator for output
        accumulator = []

        # Define log storage for information
        set_type = 'Dev' if is_development else 'Test'
        errored_path = Path(self._chkpt_path, 'error_sample_%s.log' % set_type)
        correct_path = Path(self._chkpt_path, 'correct_sample_%s.log' % set_type)
        result_path = Path(self._chkpt_path, 'results.csv')

        # Check whether we should write header or not.
        first_result_output = not result_path.exists()

        # Open file handlers
        errored_fp = errored_path.open('w+t', encoding='UTF-8')
        correct_fp = correct_path.open('w+t', encoding='UTF-8')
        result_fp = result_path.open('a+t', encoding='UTF-8')

        # Set module as evaluation phase
        self._module.eval()

        # Load dataset
        dataset = self.devset if is_development else self.evalset
        max_len = 0 if is_development else MEM_MAX
        for batch in dataset:
            # For each batch item, load it and produce outputs
            kwargs = self._load_batch(batch, is_training=False, max_len=max_len)
            outputs = self._step(**kwargs, training=False, beam=beam_size)

            # Convert text into string (for printing purpose)
            texts = dataset.problem_field.convert_ids_to_string(batch.text.token)

            # Check the result and print the result for each item.
            for i, gold, beams, results in self._check_equation(self._answer_checker, outputs, batch):
                # Record the best output of the beam search results
                result_dict = {'Index': batch.index[i],
                               'Error': str(type(results[0][2])),
                               'correct': results[0][0],
                               'error_1_Parse': results[0][2] is not None,
                               'error_2_Empty': len(results[0][1]) == 0 and results[0][2] is None,
                               'error_3_Match': not results[0][0] and len(results[0][1]) > 0 and results[0][2] is None,
                               'correct_in_beam': any(r[0] for r in results)}

                # Accumulate the test result.
                accumulator.append(result_dict)

                # Select appropriate file handler
                fp = errored_fp if not result_dict['correct'] else correct_fp
                # Write problem & result
                fp.writelines(['[Q] ', batch.index[i], '\n', texts[i], '\n',
                               '---------------------------------------\n',
                               '[EXPECTED]\t%s\n' % ' '.join(gold),
                               '---ANSWER:\t%s\n' % batch.expected[i],
                               '---------------------------------------\n'])
                fp.writelines(['[BEAM#%3d]\t%s\n'
                               '---ANSWER:\t%s\n%s' %
                               (b, ' '.join(beam), res[1],
                                '' if res[2] is None else '----ERROR:\t%s %s\n' % (type(res[2]), str(res[2])))
                               for b, (beam, res) in enumerate(zip(beams, results))])
                fp.write('\n')

        # Close file handlers
        errored_fp.close()
        correct_fp.close()

        # Write CSV results
        sorted_keys = sorted(accumulator[0].keys())
        # Write CSV header
        if first_result_output:
            _write_csv_line(result_fp, 'Set', 'GlobalStep', 'Beam', *sorted_keys)

        # Write CSV results
        for values in accumulator:
            _write_csv_line(result_fp, set_type, self._epoch, beam_size, *[values[key] for key in sorted_keys])

        # Close CSV handler
        result_fp.close()

        # Average metric across items (correctness & errors)
        metric_dict = {}
        for key in sorted_keys:
            value = [item[key] for item in accumulator]

            if type(value[0]) is not str:
                average = sum(value) / len(value)

                # Write accumulated results
                self._logger.info('Evaluating on %s (beam %s): %s = %.6f', set_type, beam_size, key, average)
                metric_dict[set_type + '/' + key] = average

        # Reset the dataset (since dataset reached EOF)
        dataset.reset()

        # Write exponential moving average & maximum value into metric dict
        if is_development:
            self._best_on_dev = max(self._best_on_dev, metric_dict['Dev/correct'])
            if self._ema_on_dev is None:
                self._ema_on_dev = metric_dict['Dev/correct']
            else:
                self._ema_on_dev = metric_dict['Dev/correct'] * 0.6 + self._ema_on_dev * 0.4

            metric_dict['Dev/correct_max'] = self._best_on_dev
            metric_dict['Dev/correct_ema'] = self._ema_on_dev

        # Record last output
        self._last_performances[set_type] = [item['correct'] for item in sorted(accumulator, key=lambda d: d['Index'])]
        self._last_metrics.update(metric_dict)


__all__ = ['Trainer']
