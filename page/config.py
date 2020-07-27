from json import load, dump as save
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoTokenizer

from page.torch.eq_field import OpEquationField, ExpressionEquationField
from page.torch.text_field import ProblemTextField
from page.util import filter_dict_by_keys
from page.const import MODEL_EXPR_PTR_TRANS


class ModelConfig:
    """
    Configuration of a model
    """
    def __init__(self, encoder_model: str = 'albert-large-v2', chkpt_path: str = None,
                 model_type: str = MODEL_EXPR_PTR_TRANS, num_decoder_layers: int = 12,
                 num_pointer_heads: int = 1, beam_size: int = 3, op_vocab_size: int = None,
                 operator_vocab_size: int = None, constant_vocab_size: int = None, operand_vocab_size: int = None):
        """
        Configuration of a model
        
        :param str encoder_model:
            Name of encoder model or path where a pre-trained model saved. 'albert-large-v2' by default. 
        :param str chkpt_path: 
            Path where the model checkpointed. None by default.
        :param str model_type: 
            Type of solver model to be built. 'ept' by default.
        :param int num_decoder_layers: 
            Number of shared decoder layers. 12 by default.
        :param int num_pointer_heads: 
            Number of pointer heads used in output layer (EPT). 1 by default (No multi-head attention)
        :param int beam_size: 
            Size of beam used in beam search. 3 by default.
        :param int op_vocab_size: 
            Size of op-token vocab.
        :param int operator_vocab_size: 
            Size of operator vocab for expression-token sequence
        :param int constant_vocab_size: 
            Size of constant vocab for pointing expression-token sequence (EPT)
        :param int operand_vocab_size: 
            Size of operand vocab for generating operands in Vanilla Transformer + Expression
        """
        self.encoder_model = encoder_model
        self.chkpt_path = chkpt_path
        self.model_type = model_type

        self.num_decoder_layers = int(num_decoder_layers)
        self.num_pointer_heads = int(num_pointer_heads)
        self.beam_size = int(beam_size)

        self.op_vocab_size = op_vocab_size
        self.operator_vocab_size = operator_vocab_size
        self.constant_vocab_size = constant_vocab_size
        self.operand_vocab_size = operand_vocab_size

        # Assign pretrained class names for runtime initialization
        self._cls = {'config': AutoConfig.from_pretrained(self.encoder_model).__class__}
        try:
            module = self._cls['config'].__module__
            clsname = self._cls['config'].__name__

            global_dict = {}
            exec('from %s import %s as tokenizer\nfrom %s import %s as model'
                 % (module.replace('configuration', 'tokenization'), clsname.replace('Config', 'Tokenizer'),
                    module.replace('configuration', 'modeling'), clsname.replace('Config', 'Model')),
                 global_dict)
            self._cls.update(global_dict)
        except:
            self._cls['tokenizer'] = AutoTokenizer.from_pretrained(self.encoder_model).__class__
            self._cls['model'] = AutoModel.from_pretrained(self.encoder_model).__class__

        self.transformer_config = self._cls['config'].from_pretrained(self.encoder_path)

    def copy(self, **kwargs):
        """
        Copy configuration with changes stated in kwargs.
        
        :keyword str encoder_model:
            Name of encoder model or path where a pre-trained model saved. 'albert-large-v2' by default. 
        :keyword str chkpt_path: 
            Path where the model checkpointed. None by default.
        :keyword str model_type: 
            Type of solver model to be built. 'ept' by default.
        :keyword int num_decoder_layers: 
            Number of shared decoder layers. 12 by default.
        :keyword int num_pointer_heads: 
            Number of pointer heads used in output layer (EPT). 1 by default (No multi-head attention)
        :keyword int beam_size: 
            Size of beam used in beam search. 3 by default.
        :keyword int op_vocab_size: 
            Size of op-token vocab.
        :keyword int operator_vocab_size: 
            Size of operator vocab for expression-token sequence
        :keyword int constant_vocab_size: 
            Size of constant vocab for pointing expression-token sequence (EPT)
        :keyword int operand_vocab_size: 
            Size of operand vocab for generating operands in Vanilla Transformer + Expression
        :rtype: ModelConfig
        :return: A copied model configuration where stated values are applied.
        """
        base = self.to_kwargs()
        for key in set(base.keys()).intersection(kwargs.keys()):
            base[key] = kwargs[key]

        return ModelConfig(**base)

    @property
    def embedding_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of input embedding
        """
        return self.transformer_config.embedding_size

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of a hidden state
        """
        return self.transformer_config.hidden_size

    @property
    def intermediate_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of intermediate feed-forward output in Transformer layers
        """
        return self.transformer_config.intermediate_size

    @property
    def num_decoder_heads(self) -> int:
        """
        :rtype: int
        :return: Number of attention heads in Transformer decoder layer
        """
        return self.transformer_config.num_attention_heads

    @property
    def init_factor(self) -> float:
        """
        :rtype: float
        :return: Initialization range used in pre-trained models
        """
        return self.transformer_config.initializer_range

    @property
    def layernorm_eps(self) -> float:
        """
        :rtype: float
        :return: Epsilon value set for Layer Normalization (for zero-safe division)
        """
        return self.transformer_config.layer_norm_eps

    @property
    def dropout_layer(self) -> float:
        """
        :rtype: float
        :return: Probability of dropout when computing hidden states
        """
        return self.transformer_config.hidden_dropout_prob

    @property
    def dropout_attn(self) -> float:
        """
        :rtype: float
        :return: Probability of dropout when computing attention scores
        """
        return self.transformer_config.attention_probs_dropout_prob

    @property
    def experiment_name(self) -> str:
        """
        :rtype: str
        :return: A string briefly explaining this model: [ModelType]-[EncoderModel]-[DecoderLayer]L[PointerHeads]P
        """
        return '%s-%s-%sL%sP' % (self.model_type, self.encoder_model.split('-')[1],
                                 self.num_decoder_layers, self.num_pointer_heads)

    @property
    def encoder_path(self) -> str:
        """
        :rtype: str
        :return: Path where pre-trained or fine-tuned encoder model saved
        """
        return self.chkpt_path if self.chkpt_path is not None else self.encoder_model

    def save_pretrained(self, path_to_save: str):
        """
        Save current configuration

        :param str path_to_save: String that represents path to the directory where this will be saved.
        """
        self.transformer_config.save_pretrained(path_to_save)

        with Path(path_to_save, 'PageConfig.json').open('w+t', encoding='UTF-8') as fp:
            save(self.to_kwargs(), fp)

    @classmethod
    def from_pretrained(cls, path: str, enforce_path: bool = False):
        """
        Load configuration

        :param str path: Path to load configuration
        :param bool enforce_path:
            True if `path` directly indicates a configuration file.
            Otherwise, system will find 'PageConfig.json' under the `path.` 
        :rtype: ModelConfig
        :return: A ModelConfig instance
        """
        path = Path(path, 'PageConfig.json') if not enforce_path else Path(path)
        with path.open('r+t', encoding='UTF-8') as fp:
            kwargs = load(fp)

        return ModelConfig(**kwargs)

    def load_encoder(self):
        """
        Load pre-trained encoder model
        
        :return: A pre-trained encoder model 
        """
        return self._cls['model'].from_pretrained(self.encoder_path)

    def load_tokenizer(self):
        """
        Load pre-trained tokenizer
        
        :return: A pre-trained tokenizer 
        """
        return self._cls['tokenizer'].from_pretrained(self.encoder_path)

    def set_chkpt_path(self, path):
        """
        Define checkpointing path
        
        :param path: Path for checkpointing  
        """
        self.chkpt_path = str(path) if path is not None else None

    def to_kwargs(self):
        """
        Change configuration as a keyword arguments
        
        :rtype: dict
        :return: Dictionary of keyword arguments to build the same configuration.
        """
        return {
            'encoder_model': self.encoder_model,
            'chkpt_path': self.chkpt_path,
            'model_type': self.model_type,
            'num_decoder_layers': self.num_decoder_layers,
            'num_pointer_heads': self.num_pointer_heads,
            'beam_size': self.beam_size,
            'op_vocab_size': self.op_vocab_size,
            'operator_vocab_size': self.operator_vocab_size,
            'constant_vocab_size': self.constant_vocab_size,
            'operand_vocab_size': self.operand_vocab_size
        }


class OptimizerConfig:
    """
    Configuration of an optimizer
    """
    
    def __init__(self, optimizer: str, **kwargs):
        """
        Configuration of an optimizer
        
        :param str optimizer:
            String representing optimizer to use.
            One of {'lamb', 'radam', 'adabound', 'yogi', 'adamw'}
        :keyword float lr: Learning rate
        :keyword float beta1: Beta 1 parameter for ADAM-based models.
        :keyword float beta2: Beta 2 parameter for ADAM-based models.
        :keyword float eps: Epsilon parameter for ADAM-based models
        :keyword float weight_decay: L2 weight decay parameter
        :keyword float clamp_value: Upper bound of weight norm (LAMB only)
        :keyword bool adam: True if enforce ADAM using LAMB code. (LAMB only)
        :keyword bool debias: True if optimizer should apply debiasing. (LAMB only)
        :keyword float final_lr: Final learning rate (ADABOUND only)
        :keyword float gamma: Gamma parameter for scheduling (ADABOUND only)
        :keyword bool amsbound: True if optimizer should use AMSBOUND (ADABOUND only)
        :keyword initial_accumulator: Initial accumulator (YOGI only) 
        """
        self.optimizer = optimizer.lower()
        kwargs['betas'] = kwargs['beta1'], kwargs['beta2']

        # Filter only acceptable keyword arguments
        if self.optimizer == 'lamb':
            kwargs = filter_dict_by_keys(kwargs,
                                         'lr', 'betas', 'eps', 'weight_decay', 'clamp_value', 'adam', 'debias')
        elif optimizer == 'radam':
            kwargs = filter_dict_by_keys(kwargs, 'lr', 'betas', 'eps', 'weight_decay')
        elif optimizer == 'adabound':
            kwargs = filter_dict_by_keys(kwargs,
                                         'lr', 'betas', 'eps', 'weight_decay', 'final_lr', 'gamma', 'amsbound')
        elif optimizer == 'yogi':
            kwargs = filter_dict_by_keys(kwargs, 'lr', 'betas', 'eps', 'weight_decay', 'initial_accumulator')
        else:  # AdamW
            kwargs = filter_dict_by_keys(kwargs, 'lr', 'betas', 'eps', 'weight_decay')

        # Enforce float or int values for non-atomic values.
        self.kwargs = {}
        for key, value in kwargs.items():
            if not isinstance(key, (int, float, str, bool, list)):
                value = float(value) if 'float' in key.dtype.name else int(value)
            self.kwargs[key] = value

    def copy(self, **kwargs):
        """
        Copy configuration with changes stated in kwargs.
        
        :keyword str optimizer:
            String representing optimizer to use.
            One of {'lamb', 'radam', 'adabound', 'yogi', 'adamw'}
        :keyword float lr: Learning rate
        :keyword float beta1: Beta 1 parameter for ADAM-based models.
        :keyword float beta2: Beta 2 parameter for ADAM-based models.
        :keyword float eps: Epsilon parameter for ADAM-based models
        :keyword float weight_decay: L2 weight decay parameter
        :keyword float clamp_value: Upper bound of weight norm (LAMB only)
        :keyword bool adam: True if enforce ADAM using LAMB code. (LAMB only)
        :keyword bool debias: True if optimizer should apply debiasing. (LAMB only)
        :keyword float final_lr: Final learning rate (ADABOUND only)
        :keyword float gamma: Gamma parameter for scheduling (ADABOUND only)
        :keyword bool amsbound: True if optimizer should use AMSBOUND (ADABOUND only)
        :keyword initial_accumulator: Initial accumulator (YOGI only) 
        :rtype: OptimizerConfig
        :return: A copied optimizer configuration where stated values are applied.
        """
        base = self.to_kwargs()
        for key in set(base.keys()).intersection(kwargs.keys()):
            base[key] = kwargs[key]

        return OptimizerConfig(**base)

    def build(self, params):
        """
        Build optimizer based on specified configuration
        
        :param params: Parameters to be trained with the new optimizer
        :return: Optimizer instance
        """
        if self.optimizer == 'lamb':
            from torch_optimizer import Lamb
            cls = Lamb
        elif self.optimizer == 'radam':
            from torch_optimizer import RAdam
            cls = RAdam
        elif self.optimizer == 'adabound':
            from torch_optimizer import AdaBound
            cls = AdaBound
        elif self.optimizer == 'yogi':
            from torch_optimizer import Yogi
            cls = Yogi
        else:
            from transformers import AdamW
            cls = AdamW

        return cls(params, **self.kwargs)

    def to_kwargs(self):
        """
        Change configuration as a keyword arguments
        
        :rtype: dict
        :return: Dictionary of keyword arguments to build the same configuration.
        """
        kwargs = self.kwargs.copy()
        kwargs['optimizer'] = self.optimizer
        kwargs['beta1'], kwargs['beta2'] = kwargs.pop('betas')

        return kwargs

    def save_pretrained(self, path_to_save: str):
        """
        Save current configuration

        :param str path_to_save: String that represents path to the directory where this will be saved.
        """
        with Path(path_to_save, 'OptConfig.json').open('w+t', encoding='UTF-8') as fp:
            save(self.to_kwargs(), fp)

    @classmethod
    def from_pretrained(cls, path: str, enforce_path: bool = False):
        """
        Load configuration

        :param str path: Path to load configuration
        :param bool enforce_path:
            True if `path` directly indicates a configuration file.
            Otherwise, system will find 'OptConfig.json' under the `path.` 
        :rtype: OptimizerConfig
        :return: A OptimizerConfig instance
        """
        path = Path(path, 'OptConfig.json') if not enforce_path else Path(path)
        with path.open('r+t', encoding='UTF-8') as fp:
            return OptimizerConfig(**load(fp))


class TrainerConfig:
    """
    Configuration of a trainer
    """
    
    def __init__(self, model: ModelConfig, optimizer: OptimizerConfig, batch: int = 4096,
                 gradient_accumulation_steps: int = 1, gradient_clip: float = 10.0, gradient_normalize: bool = False,
                 epoch: int = 1000, epoch_warmup: float = 25, epoch_chkpt: int = 10, epoch_report: int = 5,
                 fix_encoder_embedding: bool = True, lr_multiplier_encoder: float = 1.0,
                 seed: int = 1):
        """
        Configuration of a trainer
        
        :param ModelConfig model: 
            Configuration of a model to be trained 
        :param OptimizerConfig optimizer: 
            Configuration of an optimizer to be applied 
        :param int batch: 
            Size of batch in terms of tokens. 4096 by default. 
        :param int gradient_accumulation_steps: 
            Number of steps to accumulate gradients. 1 by default.
        :param float gradient_clip: 
            Clip threshold for gradient before back-propagation. 10.0 by default. 
        :param bool gradient_normalize: 
            True if a trainer should normalize gradients before applying optimizer. False by default.
        :param int epoch:
            Number of training epochs. 1000 by default. 
        :param float epoch_warmup:
            Number of warmup epochs. 25 by default 
        :param int epoch_chkpt:
            Number of epochs between checkpoints. 10 by default. 
        :param epoch_report: 
            Number of epochs between status reports. 5 by default.
        :param bool fix_encoder_embedding:
            True if a model should not update encoder's embedding matrix during the training. True by default.  
        :param float lr_multiplier_encoder:
            Multiplicative factor for adjusting encoder's learning rate separately. 1.0 by default (no adjustment) 
        :param int seed:
            Seed for RNG. 1 by default. 
        """
        self.model = model
        self.optimizer = optimizer

        self.batch = int(batch)
        self.fix_encoder_embedding = bool(fix_encoder_embedding)
        self.lr_multiplier_encoder = float(lr_multiplier_encoder)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.gradient_clip = float(gradient_clip)
        self.gradient_normalize = bool(gradient_normalize)
        self.epoch = int(epoch)
        self.epoch_warmup = float(epoch_warmup)
        self.epoch_chkpt = int(epoch_chkpt)
        self.epoch_report = int(epoch_report)
        self.seed = int(seed)

    def get(self, item):
        """
        Get given attributes by name (recursively)
        
        :param str item: Key to be find
        :return: Value of specified item
        """
        if hasattr(self, item):
            # Find item in self
            return getattr(self, item)
        else:
            # Find item in model or optimizer config
            modelargs = self.model.to_kwargs()
            if item in modelargs:
                return modelargs[item]
            else:
                return self.optimizer.to_kwargs()[item]

    def copy(self, **kwargs):
        """
        Copy configuration with changes stated in kwargs.
        
        :keyword int batch: 
            Size of batch in terms of tokens. 4096 by default. 
        :keyword int gradient_accumulation_steps: 
            Number of steps to accumulate gradients. 1 by default.
        :keyword float gradient_clip: 
            Clip threshold for gradient before back-propagation. 10.0 by default. 
        :keyword bool gradient_normalize: 
            True if a trainer should normalize gradients before applying optimizer. False by default.
        :keyword int epoch:
            Number of training epochs. 1000 by default. 
        :keyword float epoch_warmup:
            Number of warmup epochs. 25 by default 
        :keyword int epoch_chkpt:
            Number of epochs between checkpoints. 10 by default. 
        :keyword epoch_report: 
            Number of epochs between status reports. 5 by default.
        :keyword bool fix_encoder_embedding:
            True if a model should not update encoder's embedding matrix during the training. True by default.  
        :keyword float lr_multiplier_encoder:
            Multiplicative factor for adjusting encoder's learning rate separately. 1.0 by default (no adjustment) 
        :keyword int seed:
            Seed for RNG. 1 by default.
        :keyword str encoder_model:
            Name of encoder model or path where a pre-trained model saved. 'albert-large-v2' by default. 
        :keyword str chkpt_path: 
            Path where the model checkpointed. None by default.
        :keyword str model_type: 
            Type of solver model to be built. 'ept' by default.
        :keyword int num_decoder_layers: 
            Number of shared decoder layers. 12 by default.
        :keyword int num_pointer_heads: 
            Number of pointer heads used in output layer (EPT). 1 by default (No multi-head attention)
        :keyword int beam_size: 
            Size of beam used in beam search. 3 by default.
        :keyword int op_vocab_size: 
            Size of op-token vocab.
        :keyword int operator_vocab_size: 
            Size of operator vocab for expression-token sequence
        :keyword int constant_vocab_size: 
            Size of constant vocab for pointing expression-token sequence (EPT)
        :keyword int operand_vocab_size: 
            Size of operand vocab for generating operands in Vanilla Transformer + Expression
        :keyword str optimizer:
            String representing optimizer to use.
            One of {'lamb', 'radam', 'adabound', 'yogi', 'adamw'}
        :keyword float lr: Learning rate
        :keyword float beta1: Beta 1 parameter for ADAM-based models.
        :keyword float beta2: Beta 2 parameter for ADAM-based models.
        :keyword float eps: Epsilon parameter for ADAM-based models
        :keyword float weight_decay: L2 weight decay parameter
        :keyword float clamp_value: Upper bound of weight norm (LAMB only)
        :keyword bool adam: True if enforce ADAM using LAMB code. (LAMB only)
        :keyword bool debias: True if optimizer should apply debiasing. (LAMB only)
        :keyword float final_lr: Final learning rate (ADABOUND only)
        :keyword float gamma: Gamma parameter for scheduling (ADABOUND only)
        :keyword bool amsbound: True if optimizer should use AMSBOUND (ADABOUND only)
        :keyword initial_accumulator: Initial accumulator (YOGI only) 
        :rtype: TrainerConfig
        :return: A copied optimizer configuration where stated values are applied.
        """
        # Build dictionary for trainer config.
        base = dict(batch=self.batch, gradient_accumulation_steps=self.gradient_accumulation_steps,
                    gradient_clip=self.gradient_clip, gradient_normalize=self.gradient_normalize,
                    epoch=self.epoch, epoch_warmup=self.epoch_warmup, epoch_chkpt=self.epoch_chkpt,
                    epoch_report=self.epoch_report, fix_encoder_embedding=self.fix_encoder_embedding,
                    lr_multiplier_encoder=self.lr_multiplier_encoder, seed=self.seed)

        # Apply changes in trainer config
        for key in set(base.keys()).intersection(kwargs.keys()):
            base[key] = kwargs[key]

        # Apply changes model/optimizer configuration recursively.
        return TrainerConfig(self.model.copy(**kwargs), self.optimizer.copy(**kwargs), **base)

    def to_kwargs(self):
        """
        Change configuration as a keyword arguments

        :rtype: dict
        :return: Dictionary of keyword arguments to build the same configuration.
        """
        return {
            'batch': self.batch,
            'seed': self.seed,
            'fix_encoder_embedding': self.fix_encoder_embedding,
            'lr_multiplier_encoder': self.lr_multiplier_encoder,
            'gradient': {
                'accumulation_steps': self.gradient_accumulation_steps,
                'clip': self.gradient_clip,
                'normalize': self.gradient_normalize,
            },
            'epoch': {
                'total': self.epoch,
                'warmup': self.epoch_warmup,
                'chkpt': self.epoch_chkpt,
                'report': self.epoch_report,
            },
            'model': self.model.to_kwargs(),
            'optimizer': self.optimizer.to_kwargs()
        }

    def save_pretrained(self, path_to_save: str, enforce_path: bool = False):
        """
        Save current configuration

        :param str path_to_save: String that represents path to the directory where this will be saved.
        :param bool enforce_path:
            True if path_to_save specifies the target file.
            Otherwise, it will generate 'TrainConfig.json' under the specified path.
        """
        path_to_save = Path(path_to_save) if enforce_path else Path(path_to_save, 'TrainConfig.json')
        with path_to_save.open('w+t', encoding='UTF-8') as fp:
            save(self.to_kwargs(), fp)

    @classmethod
    def from_pretrained(cls, path: str):
        """
        Load configuration

        :param str path: Path to load configuration
        :rtype: TrainerConfig
        :return: A TrainerConfig instance
        """
        # Find the location of TrainConfig.json
        path = Path(path)
        if not path.is_file():
            path = Path(path, 'TrainConfig.json')
        parent = path.parent

        # Load TrainConfig
        with path.open('r+t', encoding='UTF-8') as fp:
            config = load(fp)

        # Load or generate ModelConfig
        model = config.pop('model')
        model = ModelConfig.from_pretrained(Path(parent, model), enforce_path=True) if type(model) is str \
            else ModelConfig(**model)

        # Load or generate OptimizerConfig
        optim = config.pop('optimizer')
        optim = OptimizerConfig.from_pretrained(Path(parent, optim), enforce_path=True) if type(optim) is str \
            else OptimizerConfig(**optim)

        # Read default values specified in the config.
        default = {}
        if 'default' in config:
            with Path(parent, config.pop('default')).open('r+t', encoding='UTF-8') as fp:
                default = load(fp)

        kwargs = {}
        # Apply default values first, and then overwrite specified keys with specified values
        for key, value in list(default.items()) + list(config.items()):
            if type(value) is dict:
                kwargs.update({key + ('_' + subkey if subkey else ''): val for subkey, val in value.items()})
            else:
                kwargs[key] = value

        # Rename 'epoch_total' to 'epoch'
        if 'epoch_total' in kwargs:
            kwargs['epoch'] = kwargs.pop('epoch_total')

        return TrainerConfig(model, optim, **kwargs)

    def read_datasets(self, train: str, test: str, dev: str = None):
        """
        Read datasets from the specified path.

        :param str train: JSON with lines file where training set saved
        :param str test: JSON with lines file where evaluation set saved
        :param str dev:
            JSON with lines file where development set saved (Optional)
            If unspecified, development set will automatically point evaluation set.
        :return: Triple of datasets
            - [0] Training set
            - [1] Development set
            - [2] Evaluation set
        """
        # Load Batch Iterator class
        from page.torch.dataset import TokenBatchIterator

        # Prepare fields
        tokenizer = self.model.load_tokenizer()
        prob_field = ProblemTextField(tokenizer)
        op_gen_field = OpEquationField(['X_'], ['N_'], 'C_')
        expr_gen_field = ExpressionEquationField(['X_'], ['N_'], 'C_', max_arity=2, force_generation=True)
        expr_ptr_field = ExpressionEquationField(['X_'], ['N_'], 'C_', max_arity=2, force_generation=False)

        # Load datasets
        trainset = TokenBatchIterator(train, prob_field, op_gen_field, expr_gen_field, expr_ptr_field,
                                      self.batch, testing_purpose=False)
        evalset = TokenBatchIterator(test, prob_field, op_gen_field, expr_gen_field, expr_ptr_field,
                                     self.batch, testing_purpose=True)
        if dev is not None and test != dev:
            devset = TokenBatchIterator(dev, prob_field, op_gen_field, expr_gen_field, expr_ptr_field,
                                        self.batch, testing_purpose=True)
        else:
            devset = evalset

        # Specify vocab size on the model configuration
        self.model.op_vocab_size = len(op_gen_field.token_vocab)
        self.model.operator_vocab_size = len(expr_gen_field.operator_word_vocab)
        self.model.operand_vocab_size = len(expr_gen_field.constant_word_vocab)
        self.model.constant_vocab_size = len(expr_ptr_field.constant_word_vocab)

        return trainset, devset, evalset


__all__ = ['ModelConfig', 'OptimizerConfig', 'TrainerConfig']
