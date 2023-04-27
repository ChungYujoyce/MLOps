import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return res
    
    return wrapper