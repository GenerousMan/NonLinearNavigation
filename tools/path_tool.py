import os
import sys

def get_path(path):
    f = sys._getframe()
    filename = f.f_back.f_code.co_filename
    current = os.path.dirname(filename)
    return os.path.join(current, path)