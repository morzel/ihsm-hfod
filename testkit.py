import logging


def memoize(*args, **kwargs):
    """
    Decorator which makes function cacheable. Since dict is not a hashable object we avoid
    of using kwargs with memoize for better performance

    Accepts optional parameter 'keep' which is function that implements logic to decide which value should be kept
    in memory.
    If 'keep' function called against accepted params and returned value of decorated function return True, value
    will be memozied.
    Otherwise not, so on next time will be calculated again.

    If 'keep' is None all values will be memoized.
    """
    keep = kwargs.get('keep')

    def decorator(f):
        class Memodict(dict):
            def __getitem__(self, *key):
                return dict.__getitem__(self, key)

            def __missing__(self, key):
                ret = f(*key)
                if keep is None or keep(key, ret):
                    self[key] = ret
                return ret

        return Memodict().__getitem__

    return decorator(args[0]) if args and callable(args[0]) else decorator


class _Sjutils:
    def __init__(self):
        pass

    @property
    def callable(self, *args, **kwargs):
        def _deco(func):
            def _wrapped(*args, **kwargs):
                return func(*args, **kwargs)
            return _wrapped
        return _deco

    @property
    def memoize(self, *args, **kwargs):
        return memoize(self)


sjutils = _Sjutils()

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('TESTKIT')
logger.setLevel(logging.DEBUG)

settings = {}
