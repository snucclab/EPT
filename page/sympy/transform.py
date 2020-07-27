from logging import Logger
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict, Union, Set, Optional

import sympy

from page.const import *

"""
******************
*** Exceptions ***
******************
"""


class ControlSequenceParseException(Exception):
    """
    Base class for exception occurred while parsing control sequences.
    This will be used in the outside of this file
    """
    pass


class ArgumentMissing(ControlSequenceParseException):
    """
    Raise when argument is missing for an operation
    """
    pass


class FormulaEOF(ControlSequenceParseException):
    """
    Raise when formula did not finished yet.
    """
    pass


class FormulaMissing(ControlSequenceParseException):
    """
    Raise when expression system has no formula to evaluate.
    """
    pass


class MalformedFormula(ControlSequenceParseException):
    """
    Raise when formed formula is malformed, e.g. a token assigned before [EQN]
    """
    pass


"""
*************************
*** Utility functions ***
*************************
"""


def _wrap_key_as_symbol(x: Dict[Union[str, sympy.Symbol], Union[float, int]]) -> Dict[sympy.Symbol, Union[float, int]]:
    """
    (Internal purpose) Wraps key in a dictionary as real-valued symbol.
    Instead of using this function, sympy cannot solve an expression as it thinks Symbol('x') != Symbol('x', real=True).

    :param x: a dictionary whose keys will be wrapped.
    :return: A wrapped dictionary.
    """
    return {(sympy.Symbol(k, real=True) if type(k) is str else k): v for k, v in x.items()}


def _is_constant_string(s: str) -> bool:
    """
    (Internal purpose) Check whether the string represents a constant symbol

    :param str s: string to be checked
    :rtype: bool
    :return: True if the string represents a constant symbol
    """
    return s.startswith(CON_PREFIX)


def _is_placeholder(s: str) -> bool:
    """
    (Internal purpose) Check whether the string represents a placeholder symbol:
        - number placeholder starts with N_
        - Variable starts with X_

    :param str s: string to be checked
    :rtype: bool
    :return: True if the string represents a placeholder symbol.
    """
    return s.startswith(VAR_PREFIX) or s.startswith(NUM_PREFIX)


def _is_number_symbol(sym: sympy.Expr) -> bool:
    """
    (Internal purpose) Check whether the expression represents a number:
        - number placeholder starts with N_
        - Constant values

    :param sympy.Expr sym: expression to be checked
    :rtype: bool
    :return: True if the string represents a number symbol.
    """
    name = str(sym)
    return sym.is_Symbol and (_is_constant_string(name) or name.startswith(NUM_PREFIX))


"""
***************************
*** Conversion function ***
***************************

Classes for parsing pre/postfix string notation.
"""


class ControlSequenceParser(object):
    def __init__(self, init_token: str, eos_token: str):
        self._system_of_equations = []
        self._workspace = []
        self._has_error = None
        self.init_token = init_token
        self.eos_token = eos_token

    def reset(self):
        """
        Reset the state of the parser
        """
        self._system_of_equations = []
        self._workspace = []
        self._has_error = None

    def has_error(self) -> Optional[Exception]:
        """
        :rtype: bool
        :return: `True` if an error occurred while parsing the formula.
        """
        return self._has_error

    def is_equation_formed(self):
        """
        :rtype: bool
        :return: `True` if the last equation was wrapped with a top-level relations like Equality.
        """
        raise NotImplementedError()

    def can_finish_system(self) -> bool:
        """
        :rtype: bool
        :return: `True` if the whole control sequence can be finished
        """
        return len(self._system_of_equations) > 0 or len(self._workspace) > 0

    def can_write_answer_form(self) -> bool:
        """
        :rtype: bool
        :return: `True` if it is possible to add answer formats.
        """
        return self.is_equation_formed()

    def can_add_toplevel(self, arity: int) -> bool:
        """
        :param int arity: Arity of operation to be queried.
        :rtype: bool
        :return: `True` if it is possible to append top-level 'arity'-ary relations like Equality, ...
        """
        raise NotImplementedError()

    def can_add_op(self, arity: int) -> bool:
        """
        :param int arity: Arity of operation to be queried.
        :rtype: bool
        :return: `True` if it is possible to append a 'arity'-ary operator
        """
        raise NotImplementedError()

    def can_add_variable_or_numbers(self) -> bool:
        """
        :rtype: bool
        :return: `True` if it is possible to append a number, constant, or variable symbols
        """
        raise NotImplementedError()

    def query_available_operations(self) -> Set[str]:
        """
        :rtype: Set[str]
        :return: All available operations can be used currently.
        """
        availables = set()

        for ((arity, toplv), operations) in ARITY_MAP.items():
            if (toplv and self.can_add_toplevel(arity)) or (not toplv and self.can_add_op(arity)):
                availables.update(operations)

        return availables

    def step(self, token: str):
        """
        Proceed a token to build an expression system

        :param str token: Token to append to the system.
        """
        try:
            # Iterate over control sequence.
            if token == self.init_token:
                # Reset current state
                self.reset()
            elif token == self.eos_token:
                if not self.can_finish_system():
                    raise FormulaMissing('Cannot finish this equation system!')

                # Append all formulae into the answer forms
                self._system_of_equations = self._workspace.copy()
                # Clear workspace
                self._workspace_clear()
            else:
                # Ignore other special tokens
                if _is_constant_string(token) or _is_placeholder(token):
                    if not self.can_add_variable_or_numbers():
                        raise MalformedFormula()

                    # Add constant or placeholder as symbols
                    self._workspace.append(sympy.Symbol(token, real=True))
                else:
                    self._step_operation(token)
                # Execute post-step things
                self._post_step()
        except Exception as e:
            self._has_error = e

    def steps(self, tokens: List[str]):
        """
        Proceed multiple steps of tokens to build an expression system

        :param List[str] tokens: Tokens to append to the system.
        """
        for tok in tokens:
            self.step(tok)

    def get_result(self, force=False) -> List[sympy.Expr]:
        """
        Get parsed expression system instance
        :rtype: List[sympy.Expr]
        :return: Parsing result
        """

        if not force:
            if self.has_error() is not None:
                raise self.has_error()
            if not self.can_finish_system():
                raise FormulaEOF(self._workspace)

        return self._system_of_equations

    def _step_operation(self, token: str):
        """
        Proceed a step adding an operator

        :param str token: Token represents an operator
        """
        raise NotImplementedError()

    def _post_step(self):
        """
        Postprocess states after adding a token
        """
        pass

    def _workspace_clear(self):
        """
        Clear the workspace
        """
        self._workspace.clear()


class PostfixParser(ControlSequenceParser):
    def _step_operation(self, token: str):
        op = OPERATORS[token]

        try:
            # Form a operator instance using expressions used.
            # This process can throw IndexError.
            args = []
            for _ in range(op['arity']):
                args.append(self._workspace.pop())

            # The order is reversed, since we pop arguments from a stack.
            self._workspace.append(op['convert'](*reversed(args)))
        except IndexError:
            raise ArgumentMissing()

    def is_equation_formed(self):
        # Formula formed if (1) workspace is not empty and (2) formula wrapped with a top-level relationship.
        return len(self._workspace) > 0 and self._workspace[-1].func.__name__ in TOP_LEVEL_CLASSES

    def can_add_toplevel(self, arity: int) -> bool:
        # Top-level N-ary relationship can be added if
        # (1) the last entry is not a formula yet, (2) there are exactly N argument that is free.
        return self.can_add_op(arity=arity) \
               and (len(self._workspace) <= arity
                    or self._workspace[-(arity + 1)].func.__name__ in TOP_LEVEL_CLASSES)

    def can_add_op(self, arity: int) -> bool:
        # Unary operation can be added if there are at least 1 arguments that are free.
        result = len(self._workspace) >= arity
        for arity in range(arity):
            pos = - (arity + 1)
            result = result and self._workspace[pos].func.__name__ not in TOP_LEVEL_CLASSES

        return result

    def can_add_variable_or_numbers(self) -> bool:
        # Variable and numbers can be added anytime.
        return True


class PrefixParser(ControlSequenceParser):
    def __init__(self, init_token: str, eos_token: str):
        super().__init__(init_token=init_token, eos_token=eos_token)
        self._history = []
        self._op = None
        self._op_nary = 0

    def reset(self):
        super().reset()
        self._history = []
        self._op = None
        self._op_nary = 0

    def _step_operation(self, token: str):
        op = OPERATORS[token]

        # Push information of previous operator since we met new operator.
        self._history.append((self._op, self._workspace, self._op_nary))

        # Set current operation to found operator
        self._op = op['convert']
        self._workspace = []
        self._op_nary = op['arity']

    def _post_step(self):
        # Form a expression when available, i.e. the length of workstage equals to the required number of arguments.
        while self._op is not None and len(self._workspace) == self._op_nary:
            value = self._op(*self._workspace)

            current_op, workspace, nary = self._history.pop()
            self._op = current_op
            self._workspace = workspace
            self._op_nary = nary

            self._workspace.append(value)

    def _workspace_clear(self):
        super()._workspace_clear()

        self._history = []
        self._op = None
        self._op_nary = 0

    def is_equation_formed(self):
        # An equation is formed if the history is empty.
        return len(self._history) == 0

    def can_add_toplevel(self, arity: int) -> bool:
        # A unary top-level can be added if an equation is formed, but this is not an answer format.
        return self.is_equation_formed()

    def can_add_op(self, arity: int) -> bool:
        # A n-ary operator can be added anytime except when top-level operation can be added
        return not self.is_equation_formed()

    def can_add_variable_or_numbers(self) -> bool:
        # A number can be added anytime except when top-level operation can be added
        return not self.is_equation_formed()


"""
***************************
*** Evaluation function ***
***************************

(1) Functions for evaluating Sympy-like expression.
(2) Functions for solving ExpressionSystem
(3) Class for automatic answer checking
"""


def evaluate_sympylike(sympy_eqn: sympy.Expr, numbers: List[dict] = None) -> sympy.Expr:
    """
    Evaluate sympy-like expressions and compute/simplify its value using sympy.
    Using this function, we replace dummy operations into functions in sympy.

    :param sympy.Expr sympy_eqn: Expression to be evaluated.
    :param List[dict] numbers: List of dictionaries, which contain information about the numbers
    :rtype: sympy.Expr
    :return: Expression without dummy operations.
    """
    # Replace constant placeholders.
    symbols = sympy_eqn.free_symbols
    constant_map = {}
    for item in symbols:
        if _is_constant_string(item.name):
            name = item.name[len(CON_PREFIX):]
            positive = True
            if name.endswith('_NEG'):
                positive = False
                name = name[:-4]

            if name == 'pi':
                value = sympy.pi
            elif name == 'e':
                value = sympy.E
            else:
                value = float(name.replace('_', '.'))

            constant_map[item] = value if positive else -value

    # Apply special symbols.
    constant_map.update({'pi': sympy.pi, 'e': sympy.E})

    # Substitute constants.
    sympy_eqn = sympy_eqn.subs(constant_map)

    # Replace numbers(N0, N1, ..., T0, T1, ...)
    if numbers:
        number_map = {FORMAT_NUM % i: eval(item['value']) for i, item in enumerate(numbers)}
        sympy_eqn = sympy_eqn.subs(_wrap_key_as_symbol(number_map))

    return sympy_eqn


def solve_expression_system(recv: Queue, send: Queue):
    """
    Evaluate sympy-like expressions in the expression system and solve it in Real-number domain.

    :param List[sympy.Expr] system: ExpressionSystem to be solved.
    :param List[dict] numbers: List of dictionaries, which contain information about the numbers
    :rtype: List[Dict[sympy.Expr, sympy.Expr]]
    :return: List of answer tuples.
        Note that each value in answer tuples corresponds to each expression in the answer format.
    """
    while True:
        try:
            # Receive an object
            received_object = recv.get(block=True, timeout=600)
            # Wait 600 seconds for messages
        except Empty:
            continue
        except Exception as e:
            send.put(([], e))
            continue

        if not received_object:
            # Break the loop if received_object is false.
            break

        try:
            # Note: 'Round' need some special care when using it with symbols.
            # It should be replaced after solving equations.
            system, numbers = received_object
            equations = [evaluate_sympylike(exp, numbers) for exp in system]
            candidates = sympy.solve(equations, dict=True)

            # Collect answers that satisfy given conditions.
            answers = []
            for candidate in candidates:
                if any(not v.is_real for v in candidate.values()):
                    # Ignore the solution when it is a solution with real numbers.
                    continue

                answers.append(candidate)

            send.put((answers, None))
        except Exception as e:
            send.put(([], e))

    send.close()
    recv.close()


class AnswerChecker(object):
    """
    Class for answer checking purposes.
    """

    def __init__(self, error_limit: float = 1E-1, timelimit: float = 5,
                 enable_subset_match: bool = False, is_expression_type: bool = True, logger: Logger = None):
        """
        Class for evaluating answers

        :param float error_limit:
            the maximum amount of acceptable error between the result and the answer (default 1E-1)
        :param float timelimit:
            maximum amount of allowed time for computation in seconds (default 5)
        """

        self.error_limit = error_limit
        self.time_limit = timelimit
        self.enable_subset_match = enable_subset_match

        self.solver_process = None
        self.to_solver = None
        self.from_solver = None
        self.logger = logger
        self._start_process()

        self.parser = PostfixParser(init_token=FUN_NEW_EQN if is_expression_type else SEQ_NEW_EQN,
                                    eos_token=FUN_END_EQN if is_expression_type else SEQ_END_EQN)

    def _start_process(self):
        """
        Begin child process for running sympy
        """
        try:
            recv = Queue(maxsize=4)
            send = Queue(maxsize=4)
            self.solver_process = Process(target=solve_expression_system, name='SympySolver', args=(send, recv))
            self.to_solver = send
            self.from_solver = recv
            self.solver_process.start()
        except Exception as e:
            if self.logger:
                self.logger.error('Failed to start solver process', exc_info=e)
            pass

    def close(self):
        """
        Terminate child process for sympy
        """
        try:
            child_pid = self.solver_process.pid
            self.logger.info('Sending terminate signal to solver (PID: %s)....', child_pid)
            self.to_solver.put(False)
            self.logger.info('Closing solver queues (PID: %s)....', child_pid)
            self.to_solver.close()
            self.from_solver.close()

            if self.solver_process.is_alive():
                self.logger.info('Kill the solver process (PID: %s)....', child_pid)
                self.solver_process.kill()
        except Exception as e:
            if self.logger:
                self.logger.error('Failed to kill solver process', exc_info=e)
            pass

    def _restart_process(self):
        """
        Restart child process for sympy
        """
        self.close()
        self._start_process()

    def _check_answer(self, expected: List[List[Union[float, int]]],
                      result: List[Dict[sympy.Expr, sympy.Expr]]) -> bool:
        """
        Verify whether the answer is equivalent to the obtained result.

        :param List[List[Union[float,int]]] expected:
            List of the expected answer tuples given in this problem
        :param Set[Dict[sympy.Expr,sympy.Expr]] result:
            Set of pairs obtained by evaluating the formula for this problem
        :rtype: bool
        :return: True if all the specified answers are found in one of the resulted output.
        """

        # Now, the result should not any free variables. If so, return false.
        if any(x.free_symbols for candidate in result for x in candidate.values()):
            return False

        answer_correct = []

        for ans_pair in expected:
            is_correct = False

            for candidate in result:
                # We will ignore the order of solved result.
                candidate = list(candidate.values())
                cand_in_answer = []

                for var_s in candidate:
                    for i, var_a in enumerate(ans_pair):
                        if i not in cand_in_answer and abs(var_s - var_a) < self.error_limit:
                            cand_in_answer.append(i)
                            break
                    else:
                        cand_in_answer.append(None)

                if (not self.enable_subset_match and None not in cand_in_answer) or \
                        (self.enable_subset_match and len(candidate) - cand_in_answer.count(None) == len(ans_pair)):
                    is_correct = True
                    break

            answer_correct.append(is_correct)

        # This is solvable when all the specified answers are found in one of the resulted output.
        return all(answer_correct)

    def check(self, system: List[str], numbers: List[dict], expected: List[list]):
        """
        Evalute current expression system

        :rtype: ExecutionResult
        :return: Result of execution, named tuple of (correctness, evaluated values, error type str)
        """
        try:
            self.parser.reset()
            self.parser.steps(system)
            self.parser.step(self.parser.eos_token)

            system = self.parser.get_result()
        except Exception as e:
            return False, [], e, None

        try:
            self.to_solver.put((system, numbers))
            solution, exception = self.from_solver.get(timeout=self.time_limit)
        except Empty:
            solution = []
            exception = TimeoutError()
            self.logger.info('Attempt to replace solver process since solver process is hanging...')
            self._restart_process()
        except Exception as e:
            self.logger.info('Attempt to replace solver process since pipe is broken...')
            self._restart_process()
            return False, [], e, system

        if exception:
            return False, solution, exception, system

        try:
            expected = [[var if isinstance(var, (int, float)) else eval(var) for var in ntuple] for ntuple in expected]
            correct = self._check_answer(expected, solution)

            return correct, solution, exception, system
        except Exception as exception:
            return False, solution, exception, system


__all__ = [
    'evaluate_sympylike', 'solve_expression_system',
    'ControlSequenceParser', 'PostfixParser', 'PrefixParser', 'AnswerChecker', 'ControlSequenceParseException'
]
