from math import log10
from sympy import Eq
from itertools import groupby

# Index for padding
PAD_ID = -1

# Key indices for preprocessing and field input
PREP_KEY_EQN = 0
PREP_KEY_ANS = 1
PREP_KEY_MEM = 2

# Token for text field
NUM_TOKEN = '[N]'

# String key names for inputs
IN_TXT = 'text'
IN_TPAD = 'text_pad'
IN_TNUM = 'text_num'
IN_TNPAD = 'text_numpad'
IN_EQN = 'equation'

# Dictionary of operators
OPERATORS = {
    '+': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] + x[1])},
    '-': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] - x[1])},
    '*': {'arity': 2, 'commutable': True, 'top_level': False, 'convert': (lambda *x: x[0] * x[1])},
    '/': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] / x[1])},
    '^': {'arity': 2, 'commutable': False, 'top_level': False, 'convert': (lambda *x: x[0] ** x[1])},
    '=': {'arity': 2, 'commutable': True, 'top_level': True,
          'convert': (lambda *x: Eq(x[0], x[1], evaluate=False))}
}

# Arity and top-level classes
TOP_LEVEL_CLASSES = ['Eq']
ARITY_MAP = {key: [item[-1] for item in lst]
             for key, lst in groupby(sorted([((op['arity'], op['top_level']), key) for key, op in OPERATORS.items()],
                                            key=lambda t: t[0]), key=lambda t: t[0])}

# Infinity values
NEG_INF = float('-inf')
POS_INF = float('inf')

# FOR EXPRESSION INPUT
# Token for operator field
FUN_NEW_EQN = '__NEW_EQN'
FUN_END_EQN = '__DONE'
FUN_NEW_VAR = '__NEW_VAR'
FUN_TOKENS = [FUN_NEW_EQN, FUN_END_EQN, FUN_NEW_VAR]
FUN_NEW_EQN_ID = FUN_TOKENS.index(FUN_NEW_EQN)
FUN_END_EQN_ID = FUN_TOKENS.index(FUN_END_EQN)
FUN_NEW_VAR_ID = FUN_TOKENS.index(FUN_NEW_VAR)

FUN_TOKENS_WITH_EQ = FUN_TOKENS + ['=']
FUN_EQ_SGN_ID = FUN_TOKENS_WITH_EQ.index('=')

# Token for operand field
ARG_CON = 'CONST:'
ARG_NUM = 'NUMBER:'
ARG_MEM = 'MEMORY:'
ARG_TOKENS = [ARG_CON, ARG_NUM, ARG_MEM]
ARG_CON_ID = ARG_TOKENS.index(ARG_CON)
ARG_NUM_ID = ARG_TOKENS.index(ARG_NUM)
ARG_MEM_ID = ARG_TOKENS.index(ARG_MEM)
ARG_UNK = 'UNK'
ARG_UNK_ID = 0

# Maximum capacity of variable, numbers and expression memories
VAR_MAX = 2
NUM_MAX = 32
MEM_MAX = 32

# FOR OP INPUT
SEQ_NEW_EQN = FUN_NEW_EQN
SEQ_END_EQN = FUN_END_EQN
SEQ_UNK_TOK = ARG_UNK
SEQ_TOKENS = [SEQ_NEW_EQN, SEQ_END_EQN, SEQ_UNK_TOK, '=']
SEQ_PTR_NUM = '__NUM'
SEQ_PTR_VAR = '__VAR'
SEQ_PTR_TOKENS = SEQ_TOKENS + [SEQ_PTR_NUM, SEQ_PTR_VAR]
SEQ_NEW_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_NEW_EQN)
SEQ_END_EQN_ID = SEQ_PTR_TOKENS.index(SEQ_END_EQN)
SEQ_UNK_TOK_ID = SEQ_PTR_TOKENS.index(SEQ_UNK_TOK)
SEQ_EQ_SGN_ID = SEQ_PTR_TOKENS.index('=')
SEQ_PTR_NUM_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_NUM)
SEQ_PTR_VAR_ID = SEQ_PTR_TOKENS.index(SEQ_PTR_VAR)
SEQ_GEN_NUM_ID = SEQ_PTR_NUM_ID
SEQ_GEN_VAR_ID = SEQ_GEN_NUM_ID + NUM_MAX

# Format of variable/number/expression tokens
FORMAT_VAR = 'X_%%0%dd' % (int(log10(VAR_MAX)) + 1)
FORMAT_NUM = 'N_%%0%dd' % (int(log10(NUM_MAX)) + 1)
FORMAT_MEM = 'M_%%0%dd' % (int(log10(MEM_MAX)) + 1)
VAR_PREFIX = 'X_'
NUM_PREFIX = 'N_'
CON_PREFIX = 'C_'
MEM_PREFIX = 'M_'

# Key for field names
FIELD_OP_GEN = 'op_gen'
FIELD_EXPR_GEN = 'expr_gen'
FIELD_EXPR_PTR = 'expr_ptr'

# Model names
MODEL_VANILLA_TRANS = 'vanilla'  # Vanilla Op Transformer
MODEL_EXPR_TRANS = 'expr'  # Vanilla Transformer + Expression (Expression Transformer)
MODEL_EXPR_PTR_TRANS = 'ept'  # Expression-Pointer Transformer
