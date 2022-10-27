import random

def get_random_id(length, prefix='rid'):
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    salt = ''.join(random.sample(alphabet, length))
    return '{}-{}'.format(prefix, salt)