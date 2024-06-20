import itertools
from tqdm import tqdm


def wrap_tqdm(iterable, verbose=True):
    return tqdm(iterable) if verbose else iterable


def unravel(iterable):
    return itertools.chain.from_iterable(iterable)