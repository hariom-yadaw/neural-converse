import re
import random

__author__ = 'uyaseen'


def tokenize(text):
    return [x.strip() for x in re.split('(\W+)?', text) if x.strip()]


def remove_symbols(text):
    pattern = re.compile('[^a-zA-Z ]')
    return pattern.sub('', text.lower())


# shuffle two lists
def shuffle_pair(x, y):
    data = zip(x, y)
    random.shuffle(data)
    x, y = zip(*data)
    return list(x), list(y)


# shuffle a single list
def shuffle_list(x):
    random.shuffle(x)


def shuffle_four(a, b, c, d):
    data = zip(a, b, c, d)
    random.shuffle(data)
    a, b, c, d = zip(*data)
    return list(a), list(b), list(c), list(d)
