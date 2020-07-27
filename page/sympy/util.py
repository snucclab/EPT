import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """
    Timeout Exception to handle this type of exception
    """
    pass


@contextmanager
def time_limit(seconds):
    """
    [Context] limiting time for computing an expression
    :param seconds: maximum amount of spent time for computation
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


__all__ = ['time_limit']
